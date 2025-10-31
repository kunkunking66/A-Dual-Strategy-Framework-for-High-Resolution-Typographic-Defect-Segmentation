#!/usr/bin/env python3
# experiment_architecture_comparison_fixed.py
"""
【实验一：架构对比脚本】（已修复版本）
 - 修复 GradScaler 弃用警告（使用 torch.amp.GradScaler）
 - 在 Windows 上避免 DataLoader worker 共享内存错误（num_workers=0, pin_memory=False）
 - 用 Affine 替代 ShiftScaleRotate 以消除 albumentations 警告
 - [关键修复] 为 train_loader 添加 drop_last=True，防止 DeepLabV3Plus 因批次大小为1而报错
"""
import os
import sys
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# -----------------------
# 1. 配置 (保持与主脚本一致)
# -----------------------
class CFG:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_DIR = "images/"
    MASK_DIR = "masks/"

    SEED = 42
    EPOCHS = 200  # 为快速演示可适当调低, 论文实验请使用200
    BATCH_SIZE = 6
    VALID_BATCH_SIZE = 8
    NUM_WORKERS = 4  # 在 Windows 上会被自动降为 0

    CROP_HEIGHT = 512
    CROP_WIDTH = 512
    STRIDE = 512

    MODEL_ARCH = "UnetPlusPlus"  # This will be overwritten in the loop
    ENCODER = "resnet101"
    ENCODER_WEIGHTS = "imagenet"

    CLASSES = {"background": 0, "normal_stroke": 1, "defect_area": 2}
    NUM_CLASSES = len(CLASSES)
    VALIDATION_SPLIT = 0.2
    CLASS_WEIGHTS = [0.5, 2.0, 100.0]
    WEIGHT_PATCH_DEFECT = 10.0

    TTA_SCALES = [0.75, 1.0, 1.25]
    TTA_HFLIP = True
    TTA_VFLIP = False

    MODEL_OUTPUT_PATH = "best_model_compat.pth"


# -----------------------
# Helpers, Dataset, Transforms (与主脚本完全相同)
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
        for img_path, mask_path in tqdm(zip(self.image_paths, self.mask_paths), total=len(self.image_paths),
                                        leave=False):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            # 你的原始脚本使用的是 227 -> 1, 113 -> 2
            clean_mask = np.zeros_like(mask, dtype=np.uint8)
            clean_mask[mask == 227] = 1
            clean_mask[mask == 113] = 2
            img_3ch = np.stack([img, img, img], axis=-1)
            h, w, _ = img_3ch.shape
            # 遍历切片
            for y in range(0, h - self.crop_h + 1, self.stride):
                for x in range(0, w - self.crop_w + 1, self.stride):
                    self.image_patches.append(img_3ch[y:y + self.crop_h, x:x + self.crop_w])
                    self.mask_patches.append(clean_mask[y:y + self.crop_h, x:x + self.crop_w])

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        img, mask = self.image_patches[idx], self.mask_patches[idx]
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        return img, mask.long()


def get_train_transforms():
    # 使用 Affine 替代 ShiftScaleRotate，避免 albumentations 的 FutureWarning
    return A.Compose([
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.9, 1.1), rotate=(-15, 15), shear=(-5, 5), p=0.7),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3), A.ElasticTransform(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])


def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])


# -----------------------
# 4. Model builder
# -----------------------
def build_model():
    if CFG.MODEL_ARCH == "UnetPlusPlus":
        model = smp.UnetPlusPlus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3,
                                 classes=CFG.NUM_CLASSES)
    elif CFG.MODEL_ARCH == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3,
                                  classes=CFG.NUM_CLASSES)
    elif CFG.MODEL_ARCH == "Unet":
        model = smp.Unet(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3,
                         classes=CFG.NUM_CLASSES)
    else:
        raise ValueError("Unsupported MODEL_ARCH")
    return model


# -----------------------
# 5. Training / Eval
# -----------------------
# 使用 torch.amp.GradScaler（避免 torch.cuda.amp 的弃用警告）
scaler = torch.amp.GradScaler()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    device_type = device.type if hasattr(device, "type") else "cuda"
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        # 使用 torch.amp.autocast，并根据 device 自动选择 device_type（'cuda' 或 'cpu'）
        with torch.amp.autocast(device_type=device_type):
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return running_loss / max(1, len(loader))


@torch.no_grad()
def predict_tta_for_batch(model, imgs, device, scales, hflip, vflip):
    B, C, H, W = imgs.shape
    total = torch.zeros((B, CFG.NUM_CLASSES, H, W), device=device)
    count = 0
    for s in scales:
        in_imgs = torch.nn.functional.interpolate(imgs, scale_factor=s, mode='bilinear',
                                                  align_corners=False) if s != 1.0 else imgs
        logits = model(in_imgs)
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        total += torch.softmax(logits, dim=1)
        count += 1
        if hflip:
            logits = model(torch.flip(in_imgs, dims=[3]))
            logits = torch.flip(logits, dims=[3])
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            total += torch.softmax(logits, dim=1)
            count += 1
    return total / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    total_tp = torch.zeros(CFG.NUM_CLASSES, device=device)
    total_fp = torch.zeros(CFG.NUM_CLASSES, device=device)
    total_fn = torch.zeros(CFG.NUM_CLASSES, device=device)
    pbar = tqdm(loader, desc="Valid", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits_plain = model(imgs)
        loss = loss_fn(logits_plain, masks)
        running_loss += loss.item()
        probs = predict_tta_for_batch(model, imgs, device, tuple(CFG.TTA_SCALES), CFG.TTA_HFLIP, CFG.TTA_VFLIP)
        preds = torch.argmax(probs, dim=1)
        for cls in range(CFG.NUM_CLASSES):
            total_tp[cls] += torch.sum((preds == cls) & (masks == cls))
            total_fp[cls] += torch.sum((preds == cls) & (masks != cls))
            total_fn[cls] += torch.sum((preds != cls) & (masks == cls))
    class_iou = {}
    for cls in range(CFG.NUM_CLASSES):
        denom = total_tp[cls] + total_fp[cls] + total_fn[cls] + 1e-6
        class_iou[list(CFG.CLASSES.keys())[cls]] = (total_tp[cls] / denom).item()
    return running_loss / max(1, len(loader)), class_iou


# -----------------------
# 6. Losses / Optimizer
# -----------------------
def get_loss_and_optimizer(model):
    class_weights = torch.tensor(CFG.CLASS_WEIGHTS, device=CFG.DEVICE, dtype=torch.float)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    dice = smp.losses.DiceLoss(mode="multiclass")
    focal = smp.losses.FocalLoss(mode="multiclass")

    def composite_loss(logits, target):
        return 0.5 * dice(logits, target) + 0.3 * ce(logits, target) + 0.2 * focal(logits, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return composite_loss, optimizer, scheduler


# -----------------------
# 7. 实验运行器 (Experiment Runner)
# -----------------------
def run_experiment(model_arch: str, train_img_paths, train_mask_paths, val_img_paths, val_mask_paths):
    print(f"\n{'=' * 20} 正在开始架构 '{model_arch}' 的实验 {'=' * 20}")
    CFG.MODEL_ARCH = model_arch
    CFG.MODEL_OUTPUT_PATH = f"best_model_{model_arch.lower()}.pth"
    train_ds = SequentialCropFontDataset(train_img_paths, train_mask_paths, (CFG.CROP_HEIGHT, CFG.CROP_WIDTH),
                                         CFG.STRIDE, transform=get_train_transforms())
    val_ds = SequentialCropFontDataset(val_img_paths, val_mask_paths, (CFG.CROP_HEIGHT, CFG.CROP_WIDTH), CFG.STRIDE,
                                       transform=get_valid_transforms())

    # 计算每个 patch 的采样权重：如果包含 defect_area，则权重更高
    weights = []
    for m in train_ds.mask_patches:
        try:
            contains_defect = (m == CFG.CLASSES["defect_area"]).any()
        except Exception:
            contains_defect = False
        weights.append(CFG.WEIGHT_PATCH_DEFECT if contains_defect else 1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # 针对 Windows 平台避免 shared memory / file mapping 问题
    if sys.platform.startswith("win"):
        num_workers = 0
        pin_memory = False
    else:
        # 限制 num_workers 不超过 CPU 数量 - 1
        num_workers = min(CFG.NUM_WORKERS, max(1, (os.cpu_count() or 1) - 1))
        pin_memory = True

    # ------------------- 关键修改点 -------------------
    # 添加 drop_last=True 来丢弃最后一个不完整的批次，防止BN层因batch_size=1报错
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)
    # ----------------------------------------------------
    val_loader = DataLoader(val_ds, batch_size=CFG.VALID_BATCH_SIZE, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    model = build_model().to(CFG.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_fn, optimizer, scheduler = get_loss_and_optimizer(model)
    best_defect_iou = -1.0
    best_val_iou_dict = {}

    for epoch in range(CFG.EPOCHS):
        print(f"--- [架构: {model_arch}] Epoch {epoch + 1}/{CFG.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.DEVICE)
        val_loss, val_iou = evaluate(model, val_loader, loss_fn, CFG.DEVICE)
        scheduler.step()
        defect_iou = val_iou.get("defect_area", 0.0)
        print(f"Val Loss: {val_loss:.4f} | Defect IoU: {defect_iou:.4f}")
        if defect_iou > best_defect_iou:
            best_defect_iou = defect_iou
            best_val_iou_dict = val_iou
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, CFG.MODEL_OUTPUT_PATH)
            print(f"✨ 新的最佳模型已保存至 '{CFG.MODEL_OUTPUT_PATH}' (Defect IoU = {best_defect_iou:.4f})")
    print(f"--- 架构 '{model_arch}' 实验完成 ---")
    return best_val_iou_dict


# -----------------------
# 8. 主协调器 (Main Orchestrator)
# -----------------------
def main():
    set_seed(CFG.SEED)
    print(f"实验设备: {CFG.DEVICE}")

    # ------------------- 关键修改点 -------------------
    # 调整顺序，让 DeepLabV3Plus 先执行
    architectures_to_test = ["DeepLabV3Plus", "Unet", "UnetPlusPlus"]
    # ----------------------------------------------------

    results = []
    print("正在创建固定的数据集划分...")

    # 列出图像
    all_imgs = sorted([f for f in os.listdir(CFG.IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * (1 - CFG.VALIDATION_SPLIT))
    train_names, val_names = all_imgs[:split_idx], all_imgs[split_idx:]
    train_img_paths = [os.path.join(CFG.IMAGE_DIR, n) for n in train_names]
    train_mask_paths = [os.path.join(CFG.MASK_DIR, n) for n in train_names]
    val_img_paths = [os.path.join(CFG.IMAGE_DIR, n) for n in val_names]
    val_mask_paths = [os.path.join(CFG.MASK_DIR, n) for n in val_names]
    print(f"数据集划分完毕: {len(train_names)} 训练, {len(val_names)} 验证。")

    for arch in architectures_to_test:
        best_iou_for_arch = run_experiment(
            model_arch=arch, train_img_paths=train_img_paths, train_mask_paths=train_mask_paths,
            val_img_paths=val_img_paths, val_mask_paths=val_mask_paths
        )
        # 如果 evaluate 没有返回（例如训练失败），做保护性处理
        if not best_iou_for_arch:
            best_iou_for_arch = {k: 0.0 for k in CFG.CLASSES.keys()}
        mIoU = np.mean(list(best_iou_for_arch.values()))
        best_iou_for_arch['mIoU'] = mIoU
        best_iou_for_arch['model'] = arch
        results.append(best_iou_for_arch)

    print("\n\n" + "=" * 50)
    print("          实验一：架构对比总结报告")
    print("=" * 50)
    df = pd.DataFrame(results)
    # 确保列存在顺序
    for col in ['defect_area', 'normal_stroke', 'background']:
        if col not in df.columns:
            df[col] = 0.0
    df = df[['model', 'mIoU', 'defect_area', 'normal_stroke', 'background']]
    df = df.rename(columns={
        'model': '模型架构', 'mIoU': 'mIoU', 'defect_area': 'IoU (defect)',
        'normal_stroke': 'IoU (normal)', 'background': 'IoU (bg)'
    })
    for col in ['mIoU', 'IoU (defect)', 'IoU (normal)', 'IoU (bg)']:
        df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")
    print(df.to_string(index=False))
    print("=" * 50)
    print("实验完成！以上表格可以直接用于撰写论文。")


if __name__ == "__main__":
    main()