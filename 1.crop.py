import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np

# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录

ORIGINAL_IMAGE_DIR = os.path.join(BASE_DIR, 'pic/')
ANNOTATION_DIR = os.path.join(BASE_DIR, 'annotations/')
CROPPED_CHAR_DIR = os.path.join(BASE_DIR, 'cropped_characters_resized/')

# 目标尺寸 (宽度, 高度)
TARGET_WIDTH = 64  # 你可以根据需要调整，例如 32, 48, 64, 96
TARGET_HEIGHT = 64  # 通常与TARGET_WIDTH相同
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# 填充颜色 (你的字符是黑色的，用白色填充)
PAD_COLOR_GRAYSCALE = 255  # 255 代表白色 (对于单通道灰度/二值图)
PAD_COLOR_BGR = (255, 255, 255)  # (255, 255, 255) 代表白色 (对于三通道BGR彩色图)

# 原始图片的扩展名 (如果你的图片不是.bmp，请修改这里)
IMAGE_EXTENSION = '.bmp'


# --- 辅助函数：调整图像尺寸并填充 ---
def resize_and_pad(image, target_size_tuple, pad_color_val):
    target_w, target_h = target_size_tuple
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h) if w > 0 and h > 0 else 0  # 防止除以零
    if scale == 0:  # 如果原始图像尺寸为0，则无法缩放
        # 返回一个符合目标尺寸的纯色画布
        if len(image.shape) == 2 or image.shape[2] == 1:
            pad_color_to_use = pad_color_val[0] if isinstance(pad_color_val, tuple) and len(
                pad_color_val) > 0 else pad_color_val
            return np.full((target_h, target_w), pad_color_to_use, dtype=np.uint8)
        else:
            return np.full((target_h, target_w, 3), pad_color_val, dtype=np.uint8)

    new_w, new_h = int(w * scale), int(h * scale)

    # 确保new_w和new_h至少为1，以避免cv2.resize错误
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    interpolation = cv2.INTER_CUBIC  # 或者 cv2.INTER_LANCZOS4, cv2.INTER_LINEAR

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale or single channel
        pad_color_to_use = pad_color_val[0] if isinstance(pad_color_val, tuple) and len(
            pad_color_val) > 0 else pad_color_val
        canvas = np.full((target_h, target_w), pad_color_to_use, dtype=resized_image.dtype)
        canvas[(target_h - new_h) // 2: (target_h - new_h) // 2 + new_h,
        (target_w - new_w) // 2: (target_w - new_w) // 2 + new_w] = resized_image
    else:  # Color image (BGR)
        canvas = np.full((target_h, target_w, 3), pad_color_val, dtype=resized_image.dtype)
        canvas[(target_h - new_h) // 2: (target_h - new_h) // 2 + new_h,
        (target_w - new_w) // 2: (target_w - new_w) // 2 + new_w, :] = resized_image
    return canvas


# --- 主要处理逻辑 ---
def process_all_annotations():
    if not os.path.exists(CROPPED_CHAR_DIR):
        os.makedirs(CROPPED_CHAR_DIR)
        print(f"已创建输出文件夹: {CROPPED_CHAR_DIR}")

    if not os.path.exists(ANNOTATION_DIR):
        print(f"错误: 标注文件夹 {ANNOTATION_DIR} 未找到!")
        return
    if not os.path.exists(ORIGINAL_IMAGE_DIR):
        print(f"错误: 原始图片文件夹 {ORIGINAL_IMAGE_DIR} 未找到!")
        return

    total_xml_files = 0
    processed_xml_files = 0
    total_chars_cropped = 0

    print(f"开始处理标注文件夹: {ANNOTATION_DIR}")
    print(f"原始图片文件夹: {ORIGINAL_IMAGE_DIR}")
    print(f"目标尺寸: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"输出到: {CROPPED_CHAR_DIR}\n")

    for xml_file_name in os.listdir(ANNOTATION_DIR):
        if not xml_file_name.lower().endswith('.xml'):
            continue  # 跳过非XML文件

        total_xml_files += 1
        xml_file_path = os.path.join(ANNOTATION_DIR, xml_file_name)

        # 根据XML文件名推断对应的图片文件名
        image_base_name = os.path.splitext(xml_file_name)[0]
        image_file_name = image_base_name + IMAGE_EXTENSION
        image_file_path = os.path.join(ORIGINAL_IMAGE_DIR, image_file_name)

        print(f"--- 正在尝试处理: {xml_file_name} 与 {image_file_name} ---")

        # 检查对应的图片文件是否存在
        if not os.path.exists(image_file_path):
            print(f"  错误: 图片文件 {image_file_path} 未找到! 跳过此XML。")
            continue

        # 解析XML
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
        except ET.ParseError:
            print(f"  错误: 无法解析 XML 文件 {xml_file_path}。跳过。")
            continue

        # 读取原始图片
        original_image = cv2.imread(image_file_path)
        if original_image is None:
            print(f"  错误: 无法读取图片文件 {image_file_path}。跳过。")
            continue

        # 转换为灰度图进行处理
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            image_to_process = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        elif len(original_image.shape) == 2:
            image_to_process = original_image  # 已经是单通道
        else:
            print(f"  错误: 图片 {image_file_name} 的通道数不支持: {original_image.shape}。跳过。")
            continue

        print(f"  成功读取并转换图片 (处理尺寸: {image_to_process.shape[1]}x{image_to_process.shape[0]})")

        char_count_in_this_file = 0
        for member in root.findall('object'):
            class_name_node = member.find('name')
            if class_name_node is None or not class_name_node.text:
                print("    警告: 发现一个没有名称标签的object，跳过。")
                continue
            class_name = class_name_node.text.strip()

            class_output_dir = os.path.join(CROPPED_CHAR_DIR, class_name)
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)

            bndbox_node = member.find('bndbox')
            if bndbox_node is None:
                print(f"    警告: 类别为 '{class_name}' 的object缺少bndbox信息，跳过。")
                continue

            try:
                xmin = int(float(bndbox_node.find('xmin').text))
                ymin = int(float(bndbox_node.find('ymin').text))
                xmax = int(float(bndbox_node.find('xmax').text))
                ymax = int(float(bndbox_node.find('ymax').text))
            except (ValueError, AttributeError) as e:
                print(f"    警告: 类别为 '{class_name}' 的object边界框坐标格式错误 ({e})，跳过。")
                continue

            if xmin >= xmax or ymin >= ymax:  # 确保坐标有效
                print(
                    f"    警告: 类别为 '{class_name}' 的object边界框坐标无效 (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})，跳过。")
                continue

            cropped_char = image_to_process[ymin:ymax, xmin:xmax]

            if cropped_char.size == 0:  # 检查裁剪结果是否为空
                print(f"    警告: 裁剪出的字符为空 (类别: {class_name}, 坐标: ...). 跳过。")
                continue

            current_pad_color = PAD_COLOR_GRAYSCALE  # 因为 image_to_process 是灰度图
            resized_padded_char = resize_and_pad(cropped_char, TARGET_SIZE, current_pad_color)

            # 文件名使用 image_base_name (来自XML/图片名) 来确保唯一性
            char_filename = f"{image_base_name}_{class_name}_{char_count_in_this_file}.png"
            output_path = os.path.join(class_output_dir, char_filename)

            try:
                cv2.imwrite(output_path, resized_padded_char)
                # print(f"    已保存: {output_path}") # 可以取消注释用于详细日志
                char_count_in_this_file += 1
            except Exception as e:
                print(f"    错误: 保存处理后字符 {output_path} 失败: {e}")

        if char_count_in_this_file > 0:
            print(f"  成功处理 {xml_file_name}，裁剪并保存了 {char_count_in_this_file} 个字符。\n")
            processed_xml_files += 1
            total_chars_cropped += char_count_in_this_file
        else:
            print(f"  未从 {xml_file_name} 中找到或成功处理任何字符。\n")

    print(f"--- 所有文件处理完成 ---")
    print(f"总共找到 {total_xml_files} 个XML文件。")
    print(f"成功处理了 {processed_xml_files} 个XML文件及其对应的图片。")
    print(f"总共裁剪、调整尺寸并保存了 {total_chars_cropped} 个字符图像。")
    print(f"输出保存在: {CROPPED_CHAR_DIR}")


if __name__ == '__main__':
    process_all_annotations()
