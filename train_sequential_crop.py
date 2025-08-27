import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau


# --- 1. 配置參數 (Configuration) ---
class CFG:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    IMAGE_DIR = 'images/'
    MASK_DIR = 'masks/'

    EPOCHS = 500  # 由於數據量大增，可能不需要太多 Epoch 就能收斂
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8

    # 裁剪的小圖塊尺寸
    CROP_HEIGHT = 512
    CROP_WIDTH = 512
    # 滑窗步長，等於 CROP_WIDTH 表示不重疊
    STRIDE = 512
    NUM_WORKERS = 4

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'

    CLASSES = {'background': 0, 'normal_stroke': 1, 'defect_area': 2}
    NUM_CLASSES = len(CLASSES)

    VALIDATION_SPLIT = 0.2

    MODEL_OUTPUT_PATH = "best_model_sequential_crop.pth"


# --- 2. 全新設計的數據集類，用於順序裁剪 ---
class SequentialCropFontDataset(Dataset):
    def __init__(self, image_paths, mask_paths, crop_size, stride, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.crop_h, self.crop_w = crop_size
        self.stride = stride
        self.transform = transform

        self.image_patches = []
        self.mask_patches = []

        print("正在將高分辨率圖像預先切片...")
        # 在初始化時，直接將所有大圖切成小圖塊
        for img_path, mask_path in tqdm(zip(self.image_paths, self.mask_paths), total=len(self.image_paths),
                                        desc="Preprocessing"):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            dirty_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 實時清洗
            clean_mask = np.zeros_like(dirty_mask, dtype=np.uint8)
            clean_mask[dirty_mask == 227] = 1  # Normal_Stroke
            clean_mask[dirty_mask == 113] = 2  # Defect_Area

            # 轉換為3通道
            image_3ch = np.stack([image] * 3, axis=-1)

            img_h, img_w, _ = image_3ch.shape

            # 執行滑窗切片
            for y in range(0, img_h - self.crop_h + 1, self.stride):
                for x in range(0, img_w - self.crop_w + 1, self.stride):
                    img_patch = image_3ch[y:y + self.crop_h, x:x + self.crop_w]
                    mask_patch = clean_mask[y:y + self.crop_h, x:x + self.crop_w]

                    self.image_patches.append(img_patch)
                    self.mask_patches.append(mask_patch)

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        image = self.image_patches[idx]
        mask = self.mask_patches[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()


# --- 3. 數據增強流程 ---
def get_transforms():
    # 由於裁剪已在 Dataset 類中完成，這裡不再需要 Crop 操作
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# --- 4. 訓練和驗證函數 ---
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader)


def eval_fn(loader, model, loss_fn, device, num_classes):
    model.eval()
    running_loss, total_pixels, correct_pixels = 0.0, 0, 0
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)
    loop = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            running_loss += loss.item()
            pred_labels = torch.argmax(predictions, dim=1)
            correct_pixels += torch.sum(pred_labels == masks).item()
            total_pixels += masks.nelement()
            for cls_id in range(num_classes):
                total_tp[cls_id] += torch.sum((pred_labels == cls_id) & (masks == cls_id))
                total_fp[cls_id] += torch.sum((pred_labels == cls_id) & (masks != cls_id))
                total_fn[cls_id] += torch.sum((pred_labels != cls_id) & (masks == cls_id))
    avg_loss = running_loss / len(loader)
    avg_accuracy = correct_pixels / total_pixels
    class_iou = {}
    class_names = list(CFG.CLASSES.keys())
    for cls_id in range(num_classes):
        iou = (total_tp[cls_id]) / (total_tp[cls_id] + total_fp[cls_id] + total_fn[cls_id] + 1e-6)
        class_iou[class_names[cls_id]] = iou.item()
    return avg_loss, avg_accuracy, class_iou


# --- 5. 主執行流程 ---
if __name__ == '__main__':
    print(f"Using device: {CFG.DEVICE}")

    # --- 數據準備 ---
    all_image_names = sorted([f for f in os.listdir(CFG.IMAGE_DIR) if f.endswith('.png')])
    np.random.seed(42)
    np.random.shuffle(all_image_names)
    split_idx = int(len(all_image_names) * (1 - CFG.VALIDATION_SPLIT))
    train_image_names, valid_image_names = all_image_names[:split_idx], all_image_names[split_idx:]

    train_image_paths = [os.path.join(CFG.IMAGE_DIR, name) for name in train_image_names]
    train_mask_paths = [os.path.join(CFG.MASK_DIR, name) for name in train_image_names]
    valid_image_paths = [os.path.join(CFG.IMAGE_DIR, name) for name in valid_image_names]
    valid_mask_paths = [os.path.join(CFG.MASK_DIR, name) for name in valid_image_names]

    crop_size = (CFG.CROP_HEIGHT, CFG.CROP_WIDTH)
    train_dataset = SequentialCropFontDataset(train_image_paths, train_mask_paths, crop_size, CFG.STRIDE,
                                              transform=get_transforms())
    valid_dataset = SequentialCropFontDataset(valid_image_paths, valid_mask_paths, crop_size, CFG.STRIDE,
                                              transform=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.VALID_BATCH_SIZE, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"原始訓練圖片數量: {len(train_image_names)}")
    print(f"切片後總訓練樣本數: {len(train_dataset)}")
    print(f"原始驗證圖片數量: {len(valid_image_names)}")
    print(f"切片後總驗證樣本數: {len(valid_dataset)}")

    # --- 模型準備與訓練循環 ---
    model = smp.Unet(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3,
                     classes=CFG.NUM_CLASSES).to(CFG.DEVICE)

    class_weights = torch.tensor([0.5, 2.0, 50.0], device=CFG.DEVICE)
    print(
        f"使用自定義類別權重: background={class_weights[0]}, normal_stroke={class_weights[1]}, defect_area={class_weights[2]}")

    weighted_ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')

    loss_fn = lambda pred, target: 0.5 * dice_loss(pred, target) + 0.3 * weighted_ce_loss(pred,
                                                                                          target) + 0.2 * focal_loss(
        pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_defect_iou = -1.0
    for epoch in range(CFG.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{CFG.EPOCHS} ---")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, CFG.DEVICE)
        val_loss, val_acc, val_iou_by_class = eval_fn(valid_loader, model, loss_fn, CFG.DEVICE, CFG.NUM_CLASSES)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} Results:")
        print(f"  -> Current Learning Rate: {current_lr:.6f}")
        print(f"  -> Train Loss: {train_loss:.4f}")
        print(f"  -> Val Loss: {val_loss:.4f}, Val Pixel Accuracy: {val_acc:.4f}")
        iou_report = ", ".join([f"{name}: {iou:.4f}" for name, iou in val_iou_by_class.items()])
        print(f"  -> Validation IoU -> {iou_report}")

        current_defect_iou = val_iou_by_class['defect_area']
        if current_defect_iou > best_defect_iou:
            best_defect_iou = current_defect_iou
            torch.save(model.state_dict(), CFG.MODEL_OUTPUT_PATH)
            print(f"🎉 New best model saved to '{CFG.MODEL_OUTPUT_PATH}' (Defect IoU: {best_defect_iou:.4f})")

    print("\n--- Training Finished ---")
    print(f"Best model saved at '{CFG.MODEL_OUTPUT_PATH}' with Defect Area IoU: {best_defect_iou:.4f}")