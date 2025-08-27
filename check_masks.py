import cv2
import numpy as np
import os

# --- 請修改這裡 ---
MASK_DIR = 'masks/'  # 指向您的 masks 文件夾
# ------------------

print(f"開始檢查 '{MASK_DIR}' 文件夾中的所有標注圖...")

# 獲取所有 png 文件
mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]

if not mask_files:
    print("錯誤：在 masks 文件夾中沒有找到任何 .png 文件！")
else:
    all_good = True
    for filename in mask_files:
        path = os.path.join(MASK_DIR, filename)
        try:
            # 必須以灰度模式讀取
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告：無法讀取文件 {filename}")
                continue

            # 找出所有獨一無二的像素值
            unique_values = np.unique(mask)

            # 檢查是否有不合法的值
            is_valid = all(0 <= val <= 2 for val in unique_values)

            if is_valid:
                print(f"  ✔ 文件 '{filename}' 檢查通過。像素值: {unique_values}")
            else:
                print(f"  ❌ 文件 '{filename}' 檢查失敗！發現了不合法的像素值: {unique_values}")
                all_good = False

        except Exception as e:
            print(f"處理文件 {filename} 時出錯: {e}")
            all_good = False

    print("\n--- 檢查完畢 ---")
    if not all_good:
        print("結論：您的標注圖中存在不合法的值。請按照解決方案操作。")
    else:
        print("結論：所有標注圖的像素值都在 [0, 1, 2] 範圍內，如果依然報錯，請檢查 train_mobile.py 中的 NUM_CLASSES 設置。")