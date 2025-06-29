import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import time

# --- 1. 定义超参数和设备 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TRAIN_DIR = os.path.join(BASE_DIR, 'dataset_for_training/train_reinforce/') # 你的增强后训练数据路径
# VAL_DIR = os.path.join(BASE_DIR, 'dataset_for_training/val/') # 你的验证数据路径
# 请务必修改为你的实际路径
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset_for_training/train_reinforce/')
VAL_DIR = os.path.join(BASE_DIR, 'dataset_for_training/val/')

# 检查路径是否存在
if not os.path.exists(TRAIN_DIR):
    print(f"错误: 训练数据文件夹 {TRAIN_DIR} 未找到!")
    exit()
if not os.path.exists(VAL_DIR):
    print(f"错误: 验证数据文件夹 {VAL_DIR} 未找到!")
    exit()

NUM_EPOCHS = 30  # 训练轮数，可以根据需要调整
BATCH_SIZE = 32  # 一次处理的图片数量
LEARNING_RATE = 0.001  # 学习率
IMAGE_HEIGHT = 64  # 与你数据增强时 TARGET_HEIGHT 一致
IMAGE_WIDTH = 64  # 与你数据增强时 TARGET_WIDTH 一致
NUM_CLASSES = 2  # 正常 vs 缺陷

# 检查是否有可用的GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. 数据加载与预处理 ---
# 数据转换：这里只做最基本的转换，因为我们假设图片已经是灰度图且尺寸统一
# 如果你的图片是彩色的，需要先转灰度，或者模型输入通道设为3
data_transforms = {
    'train': transforms.Compose([
        # 图片已经是灰度图，所以不需要 transforms.Grayscale()
        # 尺寸也应该在增强时统一了
        transforms.ToTensor(),  # 将 PIL Image 或 numpy.ndarray 转换为张量，并把像素值从 [0, 255] 归一化到 [0.0, 1.0]
        # transforms.Normalize([0.5], [0.5]) # 对于灰度图，单通道归一化 (均值0.5, 标准差0.5) 可选
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ]),
}

# 加载数据集
try:
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4 if DEVICE.type == 'cuda' else 0),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4 if DEVICE.type == 'cuda' else 0)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes  # 获取类别名称，例如 ['G', 'NG']

    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")
    print(f"类别名称: {class_names}")

    # 确保类别数量与 NUM_CLASSES 匹配
    if len(class_names) != NUM_CLASSES:
        print(f"错误: ImageFolder检测到的类别数量 ({len(class_names)}) 与 NUM_CLASSES ({NUM_CLASSES}) 不匹配。")
        print(f"请检查 {TRAIN_DIR} 和 {VAL_DIR} 下的子文件夹结构。")
        exit()

except FileNotFoundError:
    print("错误: 数据集文件夹未找到。请确保TRAIN_DIR和VAL_DIR路径正确，并且包含类别子文件夹。")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()


# --- 3. 定义CNN模型 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        # 输入是 IMAGE_HEIGHT x IMAGE_WIDTH 的灰度图 (1个通道)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # (64x64x1) -> (64x64x16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64x64x16) -> (32x32x16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # (32x32x16) -> (32x32x32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32x32x32) -> (16x16x32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # (16x16x32) -> (16x16x64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (16x16x64) -> (8x8x64)

        # 展平后的特征数量: 64通道 * 8高 * 8宽
        self.fc1_input_features = 64 * (IMAGE_HEIGHT // 8) * (IMAGE_WIDTH // 8)
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加Dropout防止过拟合
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.fc1_input_features)  # 展平
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
print("\n模型结构:")
print(model)

# --- 4. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()  # 适用于多分类（这里是二分类的特殊情况）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # 可选的学习率调度器

# --- 5. 训练循环 ---
def train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()  # 清零梯度

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):  # 只在训练时计算梯度
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 获取预测结果中概率最大的类别
                    loss = criterion(outputs, labels)

                    # 如果是训练阶段，则反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # if phase == 'train' and scheduler: # 如果使用了学习率调度器
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())  # .item() 将tensor转为python number
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), os.path.join(BASE_DIR, 'best_defect_model.pth'))
                print(f'Validation accuracy improved. Saved best model to best_defect_model.pth')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


# --- 开始训练 ---
print("\n开始训练...")
trained_model, history = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)


# --- 6. 评估和可视化 (可选) ---
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(BASE_DIR, 'training_curves.png'))
    plt.show()


print("\n绘制训练曲线...")
plot_training_history(history)


def evaluate_model(model, dataloader, phase='Validation'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n--- {phase} Set Evaluation ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{phase} Confusion Matrix')
    plt.savefig(os.path.join(BASE_DIR, f'{phase.lower()}_confusion_matrix.png'))
    plt.show()


print("\n在验证集上评估最终模型...")
evaluate_model(trained_model, dataloaders['val'], phase='Validation')

# 如果你有单独的测试集，也可以在这里加载并评估
# TEST_DIR = os.path.join(BASE_DIR, 'dataset_for_training/test/')
# if os.path.exists(TEST_DIR):
#     image_datasets['test'] = datasets.ImageFolder(TEST_DIR, data_transforms['val'])
#     dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if DEVICE.type == 'cuda' else 0)
#     print(f"\n在测试集上评估最终模型...")
#     evaluate_model(trained_model, dataloaders['test'], phase='Test')

print("\n训练和评估完成。最佳模型已保存为 best_defect_model.pth")
