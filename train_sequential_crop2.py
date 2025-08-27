#!/usr/bin/env python3
# train_sequential_crop_compat.py
"""
兼容 Python 3.8 + albumentations 1.4.18 的训练脚本。
功能：
 - Sequential sliding crop dataset (预切片)
 - WeightedRandomSampler（强化含缺陷 patch）
 - 可切换 Unet++ / DeepLabV3Plus（smp）
 - 数据增强（albumentations 1.4.x）
 - TTA（多尺度 + flip）用于验证指标
 - CosineAnnealingWarmRestarts scheduler
 - 多卡 support via DataParallel
 - 混合精度（torch.cuda.amp）
"""

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # 禁用 albumentations 升级提示（兼容旧环境）

import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# -----------------------
# 1. 配置
# -----------------------
class CFG:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_DIR = "images/"
    MASK_DIR = "masks/"

    SEED = 42
    EPOCHS = 200
    BATCH_SIZE = 6
    VALID_BATCH_SIZE = 8
    NUM_WORKERS = 4

    CROP_HEIGHT = 512
    CROP_WIDTH = 512
    STRIDE = 512  # 不重叠切片

    MODEL_ARCH = "UnetPlusPlus"  # 'UnetPlusPlus' or 'DeepLabV3Plus'
    ENCODER = "resnet101"
    ENCODER_WEIGHTS = "imagenet"

    # classes mapping: mask 灰度 -> label id
    CLASSES = {"background": 0, "normal_stroke": 1, "defect_area": 2}
    NUM_CLASSES = len(CLASSES)

    VALIDATION_SPLIT = 0.2

    # class weight: [bg, normal, defect]
    CLASS_WEIGHTS = [0.5, 2.0, 100.0]

    # sampler weight multiplier for patches containing defect
    WEIGHT_PATCH_DEFECT = 10.0

    # TTA for validation
    TTA_SCALES = [0.75, 1.0, 1.25]
    TTA_HFLIP = True
    TTA_VFLIP = False

    MODEL_OUTPUT_PATH = "best_model_compat.pth"


# -----------------------
# Helpers
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # speed vs determinism:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------
# 2. Dataset: 預切片
# -----------------------
class SequentialCropFontDataset(Dataset):
    def __init__(self, image_paths, mask_paths, crop_size, stride, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.crop_h, self.crop_w = crop_size
        self.stride = stride
        self.transform = transform

        self.image_patches = []
        self.mask_patches = []

        print("Pre-slicing images into patches...")
        for img_path, mask_path in tqdm(zip(self.image_paths, self.mask_paths), total=len(self.image_paths)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                print("[WARN] cannot read:", img_path, mask_path)
                continue

            # remap values -> labels (修改這裡來適應你的 mask 灰度值)
            clean_mask = np.zeros_like(mask, dtype=np.uint8)
            clean_mask[mask == 227] = 1  # normal stroke
            clean_mask[mask == 113] = 2  # defect area

            img_3ch = np.stack([img, img, img], axis=-1)
            h, w, _ = img_3ch.shape

            # slide window (no overlap by default)
            for y in range(0, h - self.crop_h + 1, self.stride):
                for x in range(0, w - self.crop_w + 1, self.stride):
                    patch_img = img_3ch[y:y + self.crop_h, x:x + self.crop_w]
                    patch_mask = clean_mask[y:y + self.crop_h, x:x + self.crop_w]
                    self.image_patches.append(patch_img)
                    self.mask_patches.append(patch_mask)

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        img = self.image_patches[idx]
        mask = self.mask_patches[idx]
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        return img, mask.long()


# -----------------------
# 3. Transforms (albumentations 1.4.x)
# -----------------------
def get_train_transforms():
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# -----------------------
# 4. Model builder
# -----------------------
def build_model():
    if CFG.MODEL_ARCH == "UnetPlusPlus":
        model = smp.UnetPlusPlus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS,
                                 in_channels=3, classes=CFG.NUM_CLASSES)
    elif CFG.MODEL_ARCH == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS,
                                  in_channels=3, classes=CFG.NUM_CLASSES)
    else:
        raise ValueError("MODEL_ARCH must be UnetPlusPlus or DeepLabV3Plus")
    return model


# -----------------------
# 5. Training / Eval (AMP + DataParallel)
# -----------------------
scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train")
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)  # raw logits (B, C, H, W)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return running_loss / max(1, len(loader))


@torch.no_grad()
def predict_tta_for_batch(model, imgs, device, scales=(1.0,), hflip=True, vflip=False):
    """
    imgs: tensor (B, C, H, W) normalized and on device
    returns: probs averaged (B, C, H, W)
    """
    B, C, H, W = imgs.shape
    total = torch.zeros((B, CFG.NUM_CLASSES, H, W), device=device)
    count = 0

    for s in scales:
        if s == 1.0:
            in_imgs = imgs
        else:
            new_h = int(H * s)
            new_w = int(W * s)
            in_imgs = torch.nn.functional.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # original
        logits = model(in_imgs)
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        total += torch.softmax(logits, dim=1)
        count += 1

        if hflip:
            fl = torch.flip(in_imgs, dims=[3])
            logits = model(fl)
            logits = torch.flip(logits, dims=[3])
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            total += torch.softmax(logits, dim=1)
            count += 1

        if vflip:
            fl = torch.flip(in_imgs, dims=[2])
            logits = model(fl)
            logits = torch.flip(logits, dims=[2])
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            total += torch.softmax(logits, dim=1)
            count += 1

    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    total_pixels = 0
    correct = 0

    num_classes = CFG.NUM_CLASSES
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)

    pbar = tqdm(loader, desc="Valid")
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # compute plain logits for loss (no augmentation) to keep loss consistent
        logits_plain = model(imgs)
        loss = loss_fn(logits_plain, masks)
        running_loss += loss.item()

        # TTA for metrics
        probs = predict_tta_for_batch(model, imgs, device, scales=tuple(CFG.TTA_SCALES), hflip=CFG.TTA_HFLIP,
                                      vflip=CFG.TTA_VFLIP)
        preds = torch.argmax(probs, dim=1)

        correct += torch.sum(preds == masks).item()
        total_pixels += masks.numel()

        for cls in range(num_classes):
            tp = torch.sum((preds == cls) & (masks == cls)).item()
            fp = torch.sum((preds == cls) & (masks != cls)).item()
            fn = torch.sum((preds != cls) & (masks == cls)).item()
            total_tp[cls] += tp
            total_fp[cls] += fp
            total_fn[cls] += fn

        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(1, len(loader))
    pixel_acc = correct / max(1, total_pixels)
    class_iou = {}
    names = list(CFG.CLASSES.keys())
    for cls in range(num_classes):
        denom = total_tp[cls] + total_fp[cls] + total_fn[cls] + 1e-6
        iou = (total_tp[cls] / denom).item()
        class_iou[names[cls]] = iou
    return avg_loss, pixel_acc, class_iou


# -----------------------
# 6. Losses / Optimizer
# -----------------------
def get_loss_and_optimizer(model):
    class_weights = torch.tensor(CFG.CLASS_WEIGHTS, device=CFG.DEVICE, dtype=torch.float)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    dice = smp.losses.DiceLoss(mode="multiclass")
    focal = smp.losses.FocalLoss(mode="multiclass")

    def composite_loss(logits, target):
        # logits: raw outputs (B,C,H,W), target: (B,H,W)
        return 0.5 * dice(logits, target) + 0.3 * ce(logits, target) + 0.2 * focal(logits, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return composite_loss, optimizer, scheduler


# -----------------------
# 7. Main
# -----------------------
def main():
    set_seed(CFG.SEED)
    print("Device:", CFG.DEVICE)

    # --- collect files ---
    all_imgs = sorted([f for f in os.listdir(CFG.IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * (1 - CFG.VALIDATION_SPLIT))
    train_names = all_imgs[:split_idx]
    val_names = all_imgs[split_idx:]

    train_img_paths = [os.path.join(CFG.IMAGE_DIR, n) for n in train_names]
    train_mask_paths = [os.path.join(CFG.MASK_DIR, n) for n in train_names]
    val_img_paths = [os.path.join(CFG.IMAGE_DIR, n) for n in val_names]
    val_mask_paths = [os.path.join(CFG.MASK_DIR, n) for n in val_names]

    # --- datasets ---
    train_ds = SequentialCropFontDataset(train_img_paths, train_mask_paths, (CFG.CROP_HEIGHT, CFG.CROP_WIDTH),
                                         CFG.STRIDE, transform=get_train_transforms())
    val_ds = SequentialCropFontDataset(val_img_paths, val_mask_paths, (CFG.CROP_HEIGHT, CFG.CROP_WIDTH), CFG.STRIDE,
                                       transform=get_valid_transforms())

    print("raw train images:", len(train_names))
    print("train patches:", len(train_ds))
    print("raw val images:", len(val_names))
    print("val patches:", len(val_ds))

    # --- build sampler to oversample defect patches ---
    weights = []
    for m in train_ds.mask_patches:
        has_defect = (m == CFG.CLASSES["defect_area"]).any()
        weights.append(CFG.WEIGHT_PATCH_DEFECT if has_defect else 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.VALID_BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # --- model ---
    model = build_model()
    model = model.to(CFG.DEVICE)

    # multi-GPU simple support
    if torch.cuda.device_count() > 1:
        print("Using DataParallel on", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    loss_fn, optimizer, scheduler = get_loss_and_optimizer(model)

    best_defect_iou = -1.0
    for epoch in range(CFG.EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{CFG.EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.DEVICE)
        val_loss, val_acc, val_iou = evaluate(model, val_loader, loss_fn, CFG.DEVICE)

        # scheduler step
        scheduler.step(epoch + 1)

        lr = optimizer.param_groups[0]["lr"]
        print(f"LR: {lr:.8f}  TrainLoss: {train_loss:.4f}  ValLoss: {val_loss:.4f}  ValAcc: {val_acc:.4f}")
        print("Val IoU:", val_iou)

        defect_iou = val_iou.get("defect_area", 0.0)
        if defect_iou > best_defect_iou:
            best_defect_iou = defect_iou
            # if using DataParallel, model.module has the real model
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, CFG.MODEL_OUTPUT_PATH)
            print("Saved best model:", CFG.MODEL_OUTPUT_PATH, " (defect IoU=", best_defect_iou, ")")

    print("Training finished. Best defect IoU:", best_defect_iou)


if __name__ == "__main__":
    main()
