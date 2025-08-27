import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os
from tqdm import tqdm


# --- 1. 配置 (Configuration) ---
class INFERENCE_CFG:
    # --- 【需要您修改】 ---
    IMAGE_PATH = "test_image_high_res.png"
    # 指向您用【隨機裁剪法】訓練出的模型權重
    MODEL_PATH = "best_model_random_crop.pth"
    OUTPUT_PATH = "result_sliding_window.png"

    # --- 【滑窗推理的核心參數】 ---
    # 窗口大小，必須與您訓練時的 CROP_SIZE 一致
    WINDOW_SIZE = 512
    # 相鄰窗口的重疊比例 (0.25 = 25% 的重疊)，可以增加以獲得更平滑的結果
    OVERLAP_RATIO = 0.25

    # --- 【模型參數，必須與訓練時完全一致】 ---
    ENCODER = 'resnet34'
    CLASSES = {'background': 0, 'normal_stroke': 1, 'defect_area': 2}
    NUM_CLASSES = len(CLASSES)
    DEFECT_CLASS_ID = CLASSES['defect_area']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_sliding_window(cfg):
    """使用滑窗法對單張高分辨率圖像進行推理和可視化"""
    print(f"使用設備: {cfg.DEVICE}")
    if not os.path.exists(cfg.IMAGE_PATH) or not os.path.exists(cfg.MODEL_PATH):
        print(f"❌ 錯誤: 找不到輸入圖片或模型文件。")
        return

    # --- 1. 加載模型 ---
    print(f"正在從 '{cfg.MODEL_PATH}' 加載模型...")
    model = smp.Unet(encoder_name=cfg.ENCODER, encoder_weights=None, in_channels=3, classes=cfg.NUM_CLASSES)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()

    # --- 2. 準備圖像和預處理流程 ---
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    original_image_for_viz = cv2.imread(cfg.IMAGE_PATH)
    gray_image_for_model = cv2.imread(cfg.IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = gray_image_for_model.shape
    input_image_3ch = np.stack([gray_image_for_model] * 3, axis=-1)

    # --- 3. 核心：滑窗推理 ---
    print("🧠 正在執行高精度滑窗推理，這可能需要一些時間...")

    # 創建一個空白的畫布，用於累加模型的預測 logits (原始分數)
    # 我們累加 logits 而不是最終的標籤，是為了更平滑的融合
    preds_full_size = np.zeros((cfg.NUM_CLASSES, img_h, img_w), dtype=np.float32)
    # 創建一個畫布，記錄每個像素被訪問 (預測) 了多少次
    visits_full_size = np.zeros((img_h, img_w), dtype=np.uint8)

    stride = int(cfg.WINDOW_SIZE * (1 - cfg.OVERLAP_RATIO))

    # tqdm 用於顯示進度條
    for y1 in tqdm(range(0, img_h, stride), desc="Sliding Window (Rows)"):
        # 確保窗口不會超出圖像底部邊界
        y2 = min(y1 + cfg.WINDOW_SIZE, img_h)
        if y2 - y1 < cfg.WINDOW_SIZE:
            y1 = max(0, y2 - cfg.WINDOW_SIZE)

        for x1 in range(0, img_w, stride):
            # 確保窗口不會超出圖像右側邊界
            x2 = min(x1 + cfg.WINDOW_SIZE, img_w)
            if x2 - x1 < cfg.WINDOW_SIZE:
                x1 = max(0, x2 - cfg.WINDOW_SIZE)

            # 裁剪出高清圖塊
            patch = input_image_3ch[y1:y2, x1:x2]

            # 預處理
            augmented = transform(image=patch)
            image_tensor = augmented['image'].unsqueeze(0).to(cfg.DEVICE)

            # 推理
            with torch.no_grad():
                patch_preds = model(image_tensor)

            # 將預測結果從 GPU 移回 CPU，並轉換為 numpy 數組
            # .squeeze(0) 移除 batch 維度, shape: (C, H, W)
            patch_preds_cpu = patch_preds.squeeze(0).cpu().numpy()

            # 累加預測 logits 和訪問次數
            preds_full_size[:, y1:y2, x1:x2] += patch_preds_cpu
            visits_full_size[y1:y2, x1:x2] += 1

    # --- 4. 結果融合與後處理 ---
    print("🧩 正在拼接和融合預測結果...")
    # 對 logits 進行平均，得到每個像素最可靠的預測分數
    avg_preds = preds_full_size / (visits_full_size + 1e-6)  # 加 1e-6 避免除以零

    # 獲取最終的預測標籤圖
    final_pred_mask = np.argmax(avg_preds, axis=0).astype(np.uint8)

    # 創建只包含缺陷區域的二值掩碼
    defect_mask = (final_pred_mask == cfg.DEFECT_CLASS_ID).astype(np.uint8)

    # --- 5. 可視化與保存 ---
    if np.sum(defect_mask) == 0:
        print("✅ 在圖片中未檢測到任何缺陷。")
        cv2.imwrite(cfg.OUTPUT_PATH, original_image_for_viz)
    else:
        print("🎨 正在標記檢測到的缺陷區域...")
        red_overlay = np.zeros_like(original_image_for_viz, dtype=np.uint8);
        red_overlay[:] = (0, 0, 255)
        red_mask = cv2.bitwise_and(red_overlay, red_overlay, mask=defect_mask)
        final_image = cv2.addWeighted(original_image_for_viz, 1.0, red_mask, 0.6, 0)
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_image, contours, -1, (0, 255, 255), 2)
        cv2.imwrite(cfg.OUTPUT_PATH, final_image)

    print(f"🎉 推理完成！高精度結果已保存至 '{cfg.OUTPUT_PATH}'")


if __name__ == '__main__':
    predict_sliding_window(INFERENCE_CFG)