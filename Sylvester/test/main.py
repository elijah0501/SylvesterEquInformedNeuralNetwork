# 导入库
import pandas as pd
import numpy as np

# 读取与展示图片
from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline

# 创建验证集
from sklearn.model_selection import train_test_split

# 评估模型
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Pytorch的相关库
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD

# 加载数据集
train = pd.read_csv('train_LbELtWX/train.csv')
test = pd.read_csv('test_ScVgIM0/test.csv')

sample_submission = pd.read_csv('sample_submission_I5njJSF.csv')

train.head()

# 加载训练图像
train_img = []
for img_name in tqdm(train['id']):
    # 定义图像路径
    image_path = 'train_LbELtWX/train/' + str(img_name) + '.png'
    # 读取图片
    img = imread(image_path, as_gray=True)
    # 归一化像素值
    img /= 255.0
    # 转换为浮点数
    img = img.astype('float32')
    # 添加到列表
    train_img.append(img)

# 转换为numpy数组
train_x = np.array(train_img)
# 定义目标
train_y = train['label'].values
train_x.shape

# 创建验证集
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
# 转换为torch张量
train_x = train_x.reshape(54000, 1, 28, 28)
train_x = torch.from_numpy(train_x)

# 转换为torch张量
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

# 训练集形状
train_x.shape, train_y.shape
# 转换为torch张量
val_x = val_x.reshape(6000, 1, 28, 28)
val_x = torch.from_numpy(val_x)

# 转换为torch张量
val_y = val_y.astype(int);
val_y = torch.from_numpy(val_y)

# 验证集形状
val_x.shape, val_y.shape


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # 定义2D卷积层
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # 定义另一个2D卷积层
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # 前项传播
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# 定义模型
model = Net()
# 定义优化器
optimizer = Adam(model.parameters(), lr=0.07)
# 定义loss函数
criterion = CrossEntropyLoss()
# 检查GPU是否可用
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

# 定义轮数
n_epochs = 25
# 空列表存储训练集损失
train_losses = []
# 空列表存储验证集损失
val_losses = []
# 训练模型
for epoch in range(n_epochs):
    train(epoch)


def train(epoch):
    model.train()
    tr_loss = 0
    # 获取训练集
    x_train, y_train = Variable(train_x), Variable(train_y)
    # 获取验证集
    x_val, y_val = Variable(val_x), Variable(val_y)
    # 转换为GPU格式
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # 清除梯度
    optimizer.zero_grad()

    # 预测训练与验证集
    output_train = model(x_train)
    output_val = model(x_val)

    # 计算训练集与验证集损失
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # 更新权重
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # 输出验证集loss
        print('Epoch : ', epoch, '\t', 'loss :', loss_val)


# 画出loss曲线
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
