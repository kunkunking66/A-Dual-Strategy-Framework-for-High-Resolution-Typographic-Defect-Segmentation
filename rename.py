import os

# 设置目录路径
directory = "pic"

# 获取目录中的所有文件
files = os.listdir(directory)

# 确保文件数量不超过52
if len(files) > 52:
    print("警告：目录中的文件数量超过52个，只会重命名前52个文件。")

# 遍历文件并重命名
for i, filename in enumerate(files[:52], start=1):
    # 构造新的文件名
    new_filename = f"IMG_{i}"

    # 获取文件的扩展名（如果需要保留扩展名）
    file_extension = os.path.splitext(filename)[1]

    # 构造完整的旧文件路径和新文件路径
    old_file_path = os.path.join(directory, filename)
    new_file_path = os.path.join(directory, new_filename + file_extension)

    # 重命名文件
    os.rename(old_file_path, new_file_path)

print("文件重命名完成！")