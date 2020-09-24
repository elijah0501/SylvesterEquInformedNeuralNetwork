import torch.nn as nn


class MatrixCNNModel(nn.Module):
    def __init__(self):
        super(MatrixCNNModel, self).__init__()

        self.cnn_layers = nn.Sequential(
            # 定义2D卷积层
            nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 定义另一个2D卷积层
            nn.Conv2d(5, 9, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(9),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 定义另一个2D卷积层
            nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2 * 2 * 16, 16),
            nn.Linear(1 * 1 * 16, 16)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
