import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import AutoAugment, AutoAugmentPolicy



# ✅ **1. 設定數據增強（Data Augmentation）**
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 隨機裁剪
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 標準化到 [-1, 1]
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ **2. 加載 CIFAR-10 數據集**
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# ✅ **3. 創建 DataLoader**
batch_size = 128  # 你可以調整 batch_size
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ✅ **2. 定義 CNN 模型**
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # 第一層卷積: 3x3, 32 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化: (32, 32) -> (16, 16)

            # 第二層卷積: 3x3, 64 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            # 第三層卷積: 3x3, 128 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #   # 池化: (16, 16) -> (8, 8)

            # 第四層卷積: 3x3, 256 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),


            # 第五層卷積: 3x3, 512 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化: (4, 4) -> (2, 2)


            # 第六層卷積: 3x3, 1024 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            # 第七層卷積: 3x3, 512 個濾波器 + BN + ReLU + 池化
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化: (2, 2) -> (1, 1)

        )

        # 全連接層: 8192 -> 512
        self.fc1 = nn.Linear(8192, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 1D 批次歸一化
        self.dropout = nn.Dropout(0.5)



        # 最後輸出層: 512 -> 10
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(x.shape[0], -1)  # Flatten

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ✅ **3. 設定訓練**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 正則化 (weight_decay)

# ✅ **4. 訓練模型**
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = correct / total

    # ✅ **5. 測試模型**
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = correct / total

    print(f'週期 {epoch + 1}, 訓練損失: {running_loss:.4f}, 訓練準確率: {train_acc:.4f}, 測試準確率: {test_acc:.4f}')

# ✅ **6. 訓練結束**
print(f'最終測試準確率: {test_acc:.4f}')
