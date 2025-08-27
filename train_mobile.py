import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# --- 1. 配置參數 (Configuration) ---
class CFG:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 數據路徑 ---
    IMAGE_DIR = 'images/'
    MASK_DIR = 'masks/'

    # --- 訓練超參數 ---
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4  # 如果GPU內存不足(out of memory)，請減小此值
    VALID_BATCH_SIZE = 8

    # --- 數據處理 ---
    IMG_HEIGHT = 512  # 根據你的需求和顯卡性能調整
    IMG_WIDTH = 512
    NUM_WORKERS = 4  # 建議設置為CPU核心數的一半左右

    # --- 模型參數 ---
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    # 我們的目標類別 (ID 必須是 0, 1, 2)
    CLASSES = {'background': 0, 'normal_stroke': 1, 'defect_area': 2}
    NUM_CLASSES = len(CLASSES)

    # --- 數據集劃分 ---
    VALIDATION_SPLIT = 0.2  # 20% 的數據作為驗證集

    # --- 輸出 ---
    MODEL_OUTPUT_PATH = "best_model_defect_iou.pth"


# --- 2. 創建數據集類 (Dataset Class) ---
class FontDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 正常讀取您的單通道灰度圖
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)

        # ------------------- 核心修改部分：動態清洗標注重 -------------------

        # 1. 讀取含有 [0, 113, 227] 灰度值的“髒”標注重
        dirty_mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 2. 準備一塊純淨的、全為 0 的畫布 (默認所有像素都是背景)
        clean_mask = np.zeros_like(dirty_mask, dtype=np.uint8)

        # 3. 根據您的 check_masks.py 輸出，定義解碼規則
        #    我們將在這裡把不標準的灰度值，映射回標準的類別 ID (0, 1, 2)

        # 將灰度值為 227 的像素 (原黃色 Normal_Stroke)，在新畫布上標記為 1
        clean_mask[dirty_mask == 227] = 1  # 映射 Normal_Stroke

        # 將灰度值為 113 的像素 (原紅色 Defect_Area)，在新畫布上標記為 2
        clean_mask[dirty_mask == 113] = 2  # 映射 Defect_Area

        # (灰度值為 0 的背景像素無需處理，因為 clean_mask 默認就是 0)

        # 4. 從現在開始，我們使用清洗乾淨的 clean_mask 進行後續所有操作
        mask = clean_mask

        # ------------------------- 修改結束 -------------------------

        # 將單通道灰度圖轉換為3通道，以適應ImageNet預訓練的骨幹網絡
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # CrossEntropyLoss 需要 Long 類型的標籤
        return image, mask.long()


# --- 3. 定義數據增強 (Data Augmentation) ---
def get_transforms(is_train=True):
    if is_train:
        # 訓練集的增強策略
        transform = A.Compose([
            A.Resize(CFG.IMG_HEIGHT, CFG.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # 驗證集只需Resize和標準化
        transform = A.Compose([
            A.Resize(CFG.IMG_HEIGHT, CFG.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return transform


# --- 4. 訓練和驗證函數 ---
def train_fn(loader, model, optimizer, loss_fn, device):
    """單個 epoch 的訓練函數"""
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def eval_fn(loader, model, loss_fn, device, num_classes):
    """單個 epoch 的驗證函數，包含完整的指標計算"""
    model.eval()
    running_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    # 初始化用於計算IoU的計數器
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)

    loop = tqdm(loader, desc="Validating")

    with torch.no_grad():
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, masks)
            running_loss += loss.item()

            # 將模型輸出轉換為預測的類別標籤 (N, H, W)
            pred_labels = torch.argmax(predictions, dim=1)

            # 計算像素準確率
            correct_pixels += torch.sum(pred_labels == masks).item()
            total_pixels += masks.nelement()

            # 為計算IoU累加每個類別的 TP, FP, FN
            for cls_id in range(num_classes):
                total_tp[cls_id] += torch.sum((pred_labels == cls_id) & (masks == cls_id))
                total_fp[cls_id] += torch.sum((pred_labels == cls_id) & (masks != cls_id))
                total_fn[cls_id] += torch.sum((pred_labels != cls_id) & (masks == cls_id))

    # 計算整個驗證集的最終指標
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

    # --- 準備數據 ---
    all_image_names = sorted([f for f in os.listdir(CFG.IMAGE_DIR) if f.endswith('.png')])
    np.random.seed(42)  # 為了可複現性
    np.random.shuffle(all_image_names)

    split_idx = int(len(all_image_names) * (1 - CFG.VALIDATION_SPLIT))
    train_image_names = all_image_names[:split_idx]
    valid_image_names = all_image_names[split_idx:]

    train_image_paths = [os.path.join(CFG.IMAGE_DIR, name) for name in train_image_names]
    train_mask_paths = [os.path.join(CFG.MASK_DIR, name) for name in train_image_names]
    valid_image_paths = [os.path.join(CFG.IMAGE_DIR, name) for name in valid_image_names]
    valid_mask_paths = [os.path.join(CFG.MASK_DIR, name) for name in valid_image_names]

    train_dataset = FontDataset(train_image_paths, train_mask_paths, transform=get_transforms(is_train=True))
    valid_dataset = FontDataset(valid_image_paths, valid_mask_paths, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.VALID_BATCH_SIZE, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(valid_dataset)} samples")

    # --- 準備模型、損失函數和優化器 ---
    model = smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=3,
        classes=CFG.NUM_CLASSES,
    ).to(CFG.DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)

    # --- 開始訓練 ---
    best_defect_iou = -1.0

    for epoch in range(CFG.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{CFG.EPOCHS} ---")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, CFG.DEVICE)
        val_loss, val_acc, val_iou_by_class = eval_fn(valid_loader, model, loss_fn, CFG.DEVICE, CFG.NUM_CLASSES)

        print(f"Epoch {epoch + 1} Results:")
        print(f"  -> Train Loss: {train_loss:.4f}")
        print(f"  -> Val Loss: {val_loss:.4f}, Val Pixel Accuracy: {val_acc:.4f}")

        # 打印每個類別的IoU
        iou_report = ", ".join([f"{name}: {iou:.4f}" for name, iou in val_iou_by_class.items()])
        print(f"  -> Validation IoU -> {iou_report}")

        # 根據缺陷區域的IoU來保存模型
        current_defect_iou = val_iou_by_class['defect_area']
        if current_defect_iou > best_defect_iou:
            best_defect_iou = current_defect_iou
            torch.save(model.state_dict(), CFG.MODEL_OUTPUT_PATH)
            print(f"🎉 New best model saved to '{CFG.MODEL_OUTPUT_PATH}' (Defect IoU: {best_defect_iou:.4f})")

    print("\n--- Training Finished ---")
    print(f"Best model saved at '{CFG.MODEL_OUTPUT_PATH}' with Defect Area IoU: {best_defect_iou:.4f}")
