#!/usr/bin/env python3
# experiment_loss_ablation.py
"""
【实验二：损失函数消融实验脚本】
目的：在固定模型架构(U-Net++)和其他所有超参数的条件下，
      对比“标准交叉熵损失(CE)”和我们提出的“复合损失(CL)”的性能差异。

功能：
 - 自动化训练流程，依次使用不同的损失函数进行训练。
 - 保证所有实验使用相同的模型架构、数据集划分和超参数。
 - 实验结束后，输出一个清晰的性能对比总结表。
"""

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

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
    STRIDE = 512
    MODEL_ARCH = "UnetPlusPlus"
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
# Helpers, Dataset, Transforms
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SequentialCropFontDataset(Dataset):
    def __init__(self, image_paths, mask_paths, crop_size, stride, transform=None):
        self.image_paths, self.mask_paths, self.crop_h, self.crop_w, self.stride, self.transform = image_paths, mask_paths, \
        crop_size[0], crop_size[1], stride, transform
        self.image_patches, self.mask_patches = [], []
        print("Pre-slicing images into patches...")
        for img_path, mask_path in tqdm(zip(self.image_paths, self.mask_paths), total=len(self.image_paths),
                                        leave=False):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None: continue
            clean_mask = np.zeros_like(mask, dtype=np.uint8)
            clean_mask[mask == 227] = 1;
            clean_mask[mask == 113] = 2
            img_3ch = np.stack([img, img, img], axis=-1)
            h, w, _ = img_3ch.shape
            for y in range(0, h - self.crop_h + 1, self.stride):
                for x in range(0, w - self.crop_w + 1, self.stride):
                    self.image_patches.append(img_3ch[y:y + self.crop_h, x:x + self.crop_w])
                    self.mask_patches.append(clean_mask[y:y + self.crop_h, x:x + self.crop_w])

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        img, mask = self.image_patches[idx], self.mask_patches[idx]
        if self.transform:
            augmented = self.transform(image=img, mask=mask);
            img, mask = augmented["image"], augmented["mask"]
        return img, mask.long()


def get_train_transforms(): return A.Compose(
    [A.ShiftScaleRotate(p=0.7), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2), A.RandomBrightnessContrast(p=0.3),
     A.ElasticTransform(p=0.2), A.Normalize(), ToTensorV2()])


def get_valid_transforms(): return A.Compose([A.Normalize(), ToTensorV2()])


# -----------------------
# 4. Model builder
# -----------------------
def build_model():
    model = smp.UnetPlusPlus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3,
                             classes=CFG.NUM_CLASSES)
    return model


# -----------------------
# 5. Training / Eval
# -----------------------
# --- 【已修复】根据PyTorch新版API修复警告 ---
# 检查CUDA是否可用，并据此决定是否启用scaler。
# 新的API `torch.amp.GradScaler()` 更通用。
scaler = torch.amp.GradScaler(enabled=(CFG.DEVICE.type == 'cuda'))


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train();
    running_loss = 0.0;
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(imgs);
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward();
        scaler.step(optimizer);
        scaler.update()
        running_loss += loss.item();
        pbar.set_postfix(loss=loss.item())
    return running_loss / max(1, len(loader))


@torch.no_grad()
def predict_tta_for_batch(model, imgs, device, scales, hflip, vflip):
    B, C, H, W = imgs.shape;
    total = torch.zeros((B, CFG.NUM_CLASSES, H, W), device=device);
    count = 0
    for s in scales:
        in_imgs = torch.nn.functional.interpolate(imgs, scale_factor=s, mode='bilinear',
                                                  align_corners=False) if s != 1.0 else imgs
        logits = model(in_imgs);
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        total += torch.softmax(logits, dim=1);
        count += 1
        if hflip:
            logits = model(torch.flip(in_imgs, dims=[3]));
            logits = torch.flip(logits, dims=[3])
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            total += torch.softmax(logits, dim=1);
            count += 1
    return total / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval();
    running_loss = 0.0
    total_tp, total_fp, total_fn = torch.zeros(CFG.NUM_CLASSES, device=device), torch.zeros(CFG.NUM_CLASSES,
                                                                                            device=device), torch.zeros(
        CFG.NUM_CLASSES, device=device)
    pbar = tqdm(loader, desc="Valid", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits_plain = model(imgs);
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
def get_loss_and_optimizer(model, loss_type='composite'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    if loss_type == 'composite':
        print("使用【复合损失函数】")
        class_weights = torch.tensor(CFG.CLASS_WEIGHTS, device=CFG.DEVICE, dtype=torch.float)
        ce = nn.CrossEntropyLoss(weight=class_weights);
        dice = smp.losses.DiceLoss(mode="multiclass");
        focal = smp.losses.FocalLoss(mode="multiclass")

        def composite_loss(logits, target):
            return 0.5 * dice(logits, target) + 0.3 * ce(logits, target) + 0.2 * focal(logits, target)

        return composite_loss, optimizer, scheduler
    elif loss_type == 'ce_only':
        print("使用【标准交叉熵损失函数】作为基线")
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn, optimizer, scheduler
    else:
        raise ValueError("loss_type 必须是 'composite' 或 'ce_only'")


# -----------------------
# 7. 实验运行器
# -----------------------
def run_experiment(loss_type: str, train_img_paths, train_mask_paths, val_img_paths, val_mask_paths):
    print(f"\n{'=' * 20} 正在开始损失函数 '{loss_type}' 的实验 {'=' * 20}")
    CFG.MODEL_OUTPUT_PATH = f"best_model_loss_{loss_type}.pth"
    train_ds = SequentialCropFontDataset(train_img_paths, train_mask_paths, (CFG.CROP_HEIGHT, CFG.CROP_WIDTH),
                                         CFG.STRIDE, transform=get_train_transforms())
    val_ds = SequentialCropFontDataset(val_img_paths, val_mask_paths, (CFG.CROP_HEIGHT, CFG.CROP_WIDTH), CFG.STRIDE,
                                       transform=get_valid_transforms())
    weights = [(CFG.WEIGHT_PATCH_DEFECT if (m == CFG.CLASSES["defect_area"]).any() else 1.0) for m in
               train_ds.mask_patches]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=CFG.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.VALID_BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS,
                            pin_memory=True)
    model = build_model().to(CFG.DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    loss_fn, optimizer, scheduler = get_loss_and_optimizer(model, loss_type=loss_type)
    best_defect_iou = -1.0;
    best_val_iou_dict = {}
    for epoch in range(CFG.EPOCHS):
        print(f"--- [损失函数: {loss_type}] Epoch {epoch + 1}/{CFG.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.DEVICE)
        val_loss, val_iou = evaluate(model, val_loader, loss_fn, CFG.DEVICE)
        scheduler.step()
        defect_iou = val_iou.get("defect_area", 0.0)
        print(f"Val Loss: {val_loss:.4f} | Defect IoU: {defect_iou:.4f}")
        if defect_iou > best_defect_iou:
            best_defect_iou = defect_iou;
            best_val_iou_dict = val_iou
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, CFG.MODEL_OUTPUT_PATH)
            print(f"✨ 新的最佳模型已保存至 '{CFG.MODEL_OUTPUT_PATH}' (Defect IoU = {best_defect_iou:.4f})")
    print(f"--- 损失函数 '{loss_type}' 实验完成 ---")
    return best_val_iou_dict


# -----------------------
# 8. 主协调器
# -----------------------
def main():
    set_seed(CFG.SEED)
    print(f"实验设备: {CFG.DEVICE}")
    loss_types_to_test = ["ce_only"]
    results = []
    print("正在创建固定的数据集划分...")
    all_imgs = sorted([f for f in os.listdir(CFG.IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * (1 - CFG.VALIDATION_SPLIT))
    train_names, val_names = all_imgs[:split_idx], all_imgs[split_idx:]
    train_img_paths = [os.path.join(CFG.IMAGE_DIR, n) for n in train_names]
    train_mask_paths = [os.path.join(CFG.MASK_DIR, n) for n in train_names]
    val_img_paths = [os.path.join(CFG.IMAGE_DIR, n) for n in val_names]
    val_mask_paths = [os.path.join(CFG.MASK_DIR, n) for n in val_names]
    print(f"数据集划分完毕: {len(train_names)} 训练, {len(val_names)} 验证。")
    for loss_type in loss_types_to_test:
        best_iou_for_loss = run_experiment(
            loss_type=loss_type, train_img_paths=train_img_paths, train_mask_paths=train_mask_paths,
            val_img_paths=val_img_paths, val_mask_paths=val_mask_paths
        )
        mIoU = np.mean(list(best_iou_for_loss.values()))
        best_iou_for_loss['mIoU'] = mIoU
        best_iou_for_loss[
            'loss_function'] = "Composite Loss [Ours]" if loss_type == 'composite' else "CE Loss [Baseline]"
        results.append(best_iou_for_loss)
    print("\n\n" + "=" * 50)
    print("        实验二：损失函数消融实验总结报告")
    print("=" * 50)
    df = pd.DataFrame(results)
    df = df[['loss_function', 'mIoU', 'defect_area', 'normal_stroke', 'background']]
    df = df.rename(columns={
        'loss_function': '损失函数', 'mIoU': 'mIoU', 'defect_area': 'IoU (defect)',
        'normal_stroke': 'IoU (normal)', 'background': 'IoU (bg)'
    })
    for col in ['mIoU', 'IoU (defect)', 'IoU (normal)', 'IoU (bg)']:
        df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")
    print(df.to_string(index=False))
    print("=" * 50)
    print("实验完成！以上表格可以直接用于撰写论文。")


if __name__ == "__main__":
    main()
