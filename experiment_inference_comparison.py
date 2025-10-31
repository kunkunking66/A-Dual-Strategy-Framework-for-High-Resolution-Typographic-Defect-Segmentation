#!/usr/bin/env python3
# experiment_inference_comparison.py
"""
【实验三：推理策略对比脚本】
目的：在同一张高分辨率图像上，使用同一个最优模型，
      直观地对比“非重叠滑窗”和“重叠滑窗”两种推理策略的视觉效果。

功能：
 - 对同一张图进行两次推理，一次重叠率为0，一次为0.25。
 - 将两次的可视化结果拼接在一张对比图上，并添加标签。
 - 【已优化】标签字体大小会根据图像宽度自适应调整，效果更美观。
 - 【再次优化】增加了标签栏高度，使文字与图像分隔更开，并微调字体大小。
"""
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# -----------------------
# 1. 配置 (请务必根据您的环境修改)
# -----------------------
class CFG:
    IMAGE_PATH = "test.png"
    MODEL_PATH = "best_model_compat.pth"  # 确保指向你最好的模型
    OUTPUT_PATH = "inference_comparison_result2.png"
    WINDOW_SIZE = 512
    OVERLAP_RATIOS_TO_TEST = (0.0, 0.5)
    MODEL_ARCH = "UnetPlusPlus"
    ENCODER = "resnet101"
    CLASSES = {"background": 0, "normal_stroke": 1, "defect_area": 2}
    NUM_CLASSES = len(CLASSES)
    DEFECT_CLASS_ID = CLASSES['defect_area']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------
# 2. 模型与图像处理函数 (与之前版本完全相同)
# -----------------------
def build_model(cfg):
    if cfg.MODEL_ARCH == "UnetPlusPlus":
        model = smp.UnetPlusPlus(encoder_name=cfg.ENCODER, encoder_weights=None, in_channels=3, classes=cfg.NUM_CLASSES)
    else:
        raise ValueError("MODEL_ARCH must be UnetPlusPlus for this experiment, or update build_model.")
    return model


def run_inference_on_image(model, original_image, transform, cfg, overlap_ratio):
    print(f"--- 正在以 {overlap_ratio * 100:.0f}% 重叠率进行推理 ---")
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray_image.shape
    input_image_3ch = np.stack([gray_image] * 3, axis=-1)
    preds_full_size = np.zeros((cfg.NUM_CLASSES, img_h, img_w), dtype=np.float32)
    visits_full_size = np.zeros((img_h, img_w), dtype=np.uint8)
    stride = int(cfg.WINDOW_SIZE * (1 - overlap_ratio)) if overlap_ratio < 1 else cfg.WINDOW_SIZE
    with torch.no_grad():
        for y1 in tqdm(range(0, img_h, stride), desc=f"Overlap {overlap_ratio * 100}%"):
            y2 = min(y1 + cfg.WINDOW_SIZE, img_h)
            if y2 - y1 < cfg.WINDOW_SIZE: y1 = max(0, y2 - cfg.WINDOW_SIZE)
            for x1 in range(0, img_w, stride):
                x2 = min(x1 + cfg.WINDOW_SIZE, img_w)
                if x2 - x1 < cfg.WINDOW_SIZE: x1 = max(0, x2 - cfg.WINDOW_SIZE)
                patch = input_image_3ch[y1:y2, x1:x2]
                image_tensor = transform(image=patch)['image'].unsqueeze(0).to(cfg.DEVICE)
                with torch.amp.autocast(device_type=cfg.DEVICE.type):
                    patch_preds = model(image_tensor)
                patch_preds_cpu = patch_preds.squeeze(0).cpu().to(torch.float32).numpy()
                preds_full_size[:, y1:y2, x1:x2] += patch_preds_cpu
                visits_full_size[y1:y2, x1:x2] += 1
    avg_preds = preds_full_size / (visits_full_size + 1e-6)
    final_pred_mask = np.argmax(avg_preds, axis=0).astype(np.uint8)
    defect_mask = (final_pred_mask == cfg.DEFECT_CLASS_ID).astype(np.uint8)
    final_image = original_image.copy()
    if np.sum(defect_mask) > 0:
        red_overlay = np.zeros_like(final_image, dtype=np.uint8);
        red_overlay[:] = (0, 0, 255)
        red_mask = cv2.bitwise_and(red_overlay, red_overlay, mask=defect_mask)
        final_image = cv2.addWeighted(final_image, 1.0, red_mask, 0.6, 0)
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_image, contours, -1, (0, 255, 255), 2)
    return final_image


# -----------------------
# 3. 主协调器 (大部分相同，只修改了第4部分)
# -----------------------
def main():
    """主函数，负责编排整个对比实验。"""
    print("=" * 50);
    print("        实验三：推理策略对比");
    print("=" * 50)
    if not os.path.exists(CFG.IMAGE_PATH): print(f"❌ 错误: 找不到测试图片 '{CFG.IMAGE_PATH}'。"); return
    if not os.path.exists(CFG.MODEL_PATH): print(f"❌ 错误: 找不到模型文件 '{CFG.MODEL_PATH}'。"); return

    # --- 1, 2, 3 部分与之前完全相同 ---
    print(f"正在从 '{CFG.MODEL_PATH}' 加载模型...")
    model = build_model(CFG)
    state_dict = torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE)
    if 'module.' in list(state_dict.keys())[0]: state_dict = {k.replace('module.', ''): v for k, v in
                                                              state_dict.items()}
    model.load_state_dict(state_dict);
    model.to(CFG.DEVICE);
    model.eval()

    original_image = cv2.imread(CFG.IMAGE_PATH)
    transform = A.Compose([A.Normalize(), ToTensorV2()])

    results_images = []
    for overlap in CFG.OVERLAP_RATIOS_TO_TEST:
        results_images.append(run_inference_on_image(model, original_image, transform, CFG, overlap))

    # --- 4. 【再次优化】创建并保存最终的、字体和间距都更优的对比图 ---
    print("\n--- 正在生成最终的优化版对比图 ---")

    h1, w1, _ = results_images[0].shape
    h2, w2, _ = results_images[1].shape
    if h1 != h2 or w1 != w2:
        target_h, target_w = min(h1, h2), min(w1, w2)
        results_images[0] = results_images[0][:target_h, :target_w]
        results_images[1] = results_images[1][:target_h, :target_w]

    comparison_body = cv2.hconcat(results_images)
    body_h, body_w, _ = comparison_body.shape

    # --- 动态计算字体和标签栏参数 ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # 黑色字体

    text1 = f"Non-Overlapping (Overlap = {CFG.OVERLAP_RATIOS_TO_TEST[0] * 100:.0f}%)"
    available_width_for_text = body_w / 2

    # 【修改】目标比例从0.8提高到0.9，让字体更大
    target_text_width = available_width_for_text * 0.9

    (base_text_width, base_text_height), _ = cv2.getTextSize(text1, font, 1, 1)
    font_scale = target_text_width / base_text_width
    font_thickness = max(1, int(font_scale * 1.5))  # 粗细也随之调整

    (final_text_width, final_text_height), _ = cv2.getTextSize(text1, font, font_scale, font_thickness)

    # 【修改】增加标签栏高度的边距，从上下各20(总共40)增加到上下各40(总共80)
    header_h = final_text_height + 80
    header = np.ones((header_h, body_w, 3), dtype=np.uint8) * 255

    # --- 在标签栏上绘制文字 ---
    pos1_x = int((available_width_for_text / 2) - (final_text_width / 2))
    pos1_y = int((header_h + final_text_height) / 2)
    cv2.putText(header, text1, (pos1_x, pos1_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    text2 = f"Overlapping (Overlap = {CFG.OVERLAP_RATIOS_TO_TEST[1] * 100:.0f}%)"
    (text2_width, _), _ = cv2.getTextSize(text2, font, font_scale, font_thickness)
    pos2_x = int(available_width_for_text + (available_width_for_text / 2) - (text2_width / 2))
    pos2_y = int((header_h + final_text_height) / 2)
    cv2.putText(header, text2, (pos2_x, pos2_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    final_comparison_image = cv2.vconcat([header, comparison_body])
    cv2.imwrite(CFG.OUTPUT_PATH, final_comparison_image)

    print("=" * 50);
    print(f"🎉 实验完成！优化版对比图已保存至 '{CFG.OUTPUT_PATH}'");
    print("=" * 50)


if __name__ == '__main__':
    main()