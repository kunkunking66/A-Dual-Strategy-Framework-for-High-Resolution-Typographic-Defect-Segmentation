import cv2
import os


def batch_convert_to_grayscale(input_dir='pic', output_dir='image'):
    """
    批量將指定輸入文件夾中的所有圖像轉換為灰度圖，
    並保存到指定的輸出文件夾中。

    Args:
        input_dir (str): 包含原始彩色圖像的文件夾名稱。
        output_dir (str): 用於存儲處理後灰度圖像的文件夾名稱。
    """
    # 檢查輸入文件夾是否存在
    if not os.path.isdir(input_dir):
        print(f"錯誤：輸入文件夾 '{input_dir}' 不存在。")
        print("請創建 'pic' 文件夾並將圖片放入其中。")
        return

    # 檢查並創建輸出文件夾
    if not os.path.exists(output_dir):
        print(f"輸出文件夾 '{output_dir}' 不存在，正在創建...")
        os.makedirs(output_dir)
        print(f"文件夾 '{output_dir}' 創建成功。")

    # 支持的圖片文件擴展名
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # 獲取輸入文件夾中所有文件名
    files_to_process = os.listdir(input_dir)

    if not files_to_process:
        print(f"警告：輸入文件夾 '{input_dir}' 為空，沒有可處理的圖片。")
        return

    print(f"\n開始處理 '{input_dir}' 文件夾中的圖片...")
    processed_count = 0
    skipped_count = 0

    # 遍歷所有文件
    for filename in files_to_process:
        # 檢查文件是否為支持的圖片格式
        if filename.lower().endswith(supported_extensions):
            try:
                # 構造完整的文件路徑
                input_path = os.path.join(input_dir, filename)

                # 讀取圖像
                image = cv2.imread(input_path)

                # 如果圖片讀取失敗，則跳過
                if image is None:
                    print(f"  - 警告：無法讀取圖片 '{filename}'，已跳過。")
                    skipped_count += 1
                    continue

                # 將圖像轉換為灰度圖
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 構造新的文件名 (例如: 'image.jpg' -> 'image_grey.jpg')
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_grey{ext}"

                # 構造完整的輸出路徑
                output_path = os.path.join(output_dir, new_filename)

                # 保存灰度圖像
                cv2.imwrite(output_path, gray_image)

                print(f"  ✔ 成功處理 '{filename}' -> 已保存至 '{output_path}'")
                processed_count += 1

            except Exception as e:
                print(f"  - 錯誤：處理文件 '{filename}' 時發生意外: {e}")
                skipped_count += 1
        else:
            # 如果不是支持的圖片格式，則跳過
            print(f"  - 提示：文件 '{filename}' 不是支持的圖片格式，已跳過。")
            skipped_count += 1

    print("\n--------------------")
    print("所有文件處理完畢。")
    print(f"成功處理: {processed_count} 個文件")
    print(f"跳過/失敗: {skipped_count} 個文件")
    print(f"灰度圖像已全部保存在 '{output_dir}' 文件夾中。")
    print("--------------------")


if __name__ == '__main__':
    # 直接運行此腳本即可
    batch_convert_to_grayscale()