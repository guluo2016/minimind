import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import numpy as np
from PIL import Image

# 检查是否有GPU可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据处理 ---
# 定义数据转换：将图片转换为张量并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

def show():
    plt.figure(figsize=(16, 9))
    for i, item in enumerate(itertools.islice(train_loader,2,102)):
        plt.subplot(10, 10, i+1)
        img,label= item
        img = img[0].cpu().numpy()
        array = (img.reshape((28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(array, 'L')
        label = label.cpu().numpy()[0]
        plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

show()

# --- 2. 模型架构 ---
# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 输入通道1，输出通道16，卷积核5x5
        # in_channels=1，表示是是单通道的灰色图像，如果是彩色的RGB图像，则是in_channels=3
        # out_channels=16，表示卷积层输出16个特征图
        # kernel_size=5，表示卷积核的大小为5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)

        # 卷积层 2: 输入通道16，输出通道32，卷积核5x5
        # 第一层已经定义了16个输出通道，因此这里接收的输入通道数是16，需要与上一层的输出通道数一致
        # out_channels=32，表示卷积层输出32个特征图
        # kernel_size=5，表示卷积核的大小为5x5
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)

        # 全连接层: 将特征图展平后输入，最终输出10个类别 (0-9)
        # in_features=32 * 20 * 20，表示输入特征的数量
        # 之所以是32 * 20 * 20，是因为经过第一层神经网络后，特征图尺寸从28x28变为24x24（28-5+1），经过第二层神经网络后，特征图尺寸从24x24变为20x20（24-5+1），而第二层神经网络模型又定义了输出通道out_channels=32，因此每个特征图有32个通道
        #  每个特征图有32个通道，因此总的输入特征数量是32 * 20 * 20
        # out_features=10，表示输出的类别数量，即数字0-9共10类
        self.fc = nn.Linear(in_features=32 * 20 * 20, out_features=10)

    def forward(self, x):
        # 激活函数使用ReLU
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        # 没有池化，展平后的特征图尺寸为 32×20×20
        x = x.view(-1, 32 * 20 * 20)

        # 全连接层
        x = self.fc(x)
        return x
    
# 实例化模型
model = SimpleCNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. 训练循环 ---
def train(model, device, train_loader, optimizer, epoch):
    model.train() # 设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad() # 梯度清零
        output = model(data)
        loss = criterion(output, target)
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# --- 4. 测试循环 ---
def test(model, device, test_loader):
    model.eval() # 设置为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 在测试阶段关闭梯度计算，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # 找到概率最大的类别
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 运行训练和测试
if __name__ == '__main__':
    epochs = 5
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)