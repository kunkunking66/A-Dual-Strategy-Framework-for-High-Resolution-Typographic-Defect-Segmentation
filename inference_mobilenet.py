import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os


# --- 1. 配置 (Configuration) ---
class INFERENCE_CFG:
    # --- 【需要您修改】 ---
    # 指向您的高分辨率灰度測試圖
    IMAGE_PATH = "test_image_high_res.png"

    # 【重要修改1】: 指向您用 MobileNetV2 訓練出的模型權重
    MODEL_PATH = "best_model_defect_iou.pth"

    # 指定一個新的輸出文件名，以區分結果
    OUTPUT_PATH = "result_from_mobilenet.png"

    # --- 【重要修改2】: 必須與您第一個模型的訓練配置完全一致 ---
    # 您第一個模型使用的骨幹是 mobilenet_v2
    ENCODER = 'mobilenet_v2'

    # --- 【以下部分保持不變】 ---
    CLASSES = {'background': 0, 'normal_stroke': 1, 'defect_area': 2}
    NUM_CLASSES = len(CLASSES)
    DEFECT_CLASS_ID = CLASSES['defect_area']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INFERENCE_SIZE = 768


def predict_and_visualize(cfg):
    """加載模型並對單張高分辨率灰度圖進行推理和可視化"""
    print(f"使用設備: {cfg.DEVICE}")
    print(f"========== 正在測試 MobileNetV2 模型 ==========")

    # --- 檢查文件是否存在 ---
    if not os.path.exists(cfg.IMAGE_PATH):
        print(f"❌ 錯誤: 找不到輸入圖片 '{cfg.IMAGE_PATH}'")
        return
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"❌ 錯誤: 找不到模型文件 '{cfg.MODEL_PATH}'")
        return

    # --- 1. 加載模型 (架構必須匹配) ---
    print(f"正在從 '{cfg.MODEL_PATH}' 加載模型...")
    model = smp.Unet(
        encoder_name=cfg.ENCODER,  # 使用 mobilenet_v2 架構
        encoder_weights=None,
        in_channels=3,
        classes=cfg.NUM_CLASSES,
    )
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()

    # --- 2. 圖像預處理 (保持不變) ---
    transform = A.Compose([
        A.Resize(cfg.INFERENCE_SIZE, cfg.INFERENCE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    original_image_for_viz = cv2.imread(cfg.IMAGE_PATH)
    original_h, original_w, _ = original_image_for_viz.shape

    gray_image_for_model = cv2.imread(cfg.IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    input_image = np.stack([gray_image_for_model] * 3, axis=-1)

    augmented = transform(image=input_image)
    image_tensor = augmented['image'].unsqueeze(0).to(cfg.DEVICE)

    print("🧠 正在執行推理...")
    # --- 3. 執行推理 (保持不變) ---
    with torch.no_grad():
        predictions = model(image_tensor)

    # --- 4. 後處理和可視化 (保持不變) ---
    pred_mask_resized = torch.argmax(predictions.squeeze(0), dim=0).cpu().numpy()
    defect_mask_resized = (pred_mask_resized == cfg.DEFECT_CLASS_ID).astype(np.uint8)
    defect_mask_original_size = cv2.resize(
        defect_mask_resized, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    if np.sum(defect_mask_original_size) == 0:
        print("✅ 在圖片中未檢測到任何缺陷。")
        cv2.imwrite(cfg.OUTPUT_PATH, original_image_for_viz)
        print(f"結果已保存至 '{cfg.OUTPUT_PATH}'")
        return

    print("🎨 正在標記檢測到的缺陷區域...")
    red_overlay = np.zeros_like(original_image_for_viz, dtype=np.uint8);
    red_overlay[:] = (0, 0, 255)
    red_mask = cv2.bitwise_and(red_overlay, red_overlay, mask=defect_mask_original_size)
    final_image = cv2.addWeighted(original_image_for_viz, 1.0, red_mask, 0.6, 0)
    contours, _ = cv2.findContours(defect_mask_original_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_image, contours, -1, (0, 255, 255), 2)

    # --- 5. 保存結果 ---
    cv2.imwrite(cfg.OUTPUT_PATH, final_image)
    print(f"🎉 推理完成！結果已保存至 '{cfg.OUTPUT_PATH}'")


if __name__ == '__main__':
    predict_and_visualize(INFERENCE_CFG)