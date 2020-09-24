from networkx.drawing.tests.test_pylab import plt
from torch.utils.data import DataLoader
from torchvision import transforms

from NormalMatrixDataset import CustomDatasetFromMatrix
from SlyvesterDataset import CustomDatasetFromImage


def LoadData(trainset_path, testset_path, val_path):
    # 定义transforms
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        # 归一化到（-1，1）
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 换数据集修改
    # 加载图片数据
    trainset = CustomDatasetFromImage(trainset_path, trans)
    testset = CustomDatasetFromImage(testset_path, trans)
    valset = CustomDatasetFromImage(val_path, trans)

    # 加载矩阵数据
    # trainset = CustomDatasetFromMatrix(trainset_path, trans)
    # testset = CustomDatasetFromMatrix(testset_path, trans)
    # valset = CustomDatasetFromMatrix(val_path, trans)

    # 加载进DataLoader
    traindata = DataLoader(dataset=trainset, batch_size=100, shuffle=True, drop_last=False)
    testdata = DataLoader(dataset=testset, batch_size=100, shuffle=True, drop_last=True)
    valdata = DataLoader(dataset=valset, batch_size=100, shuffle=True, drop_last=True)

    return traindata, testdata, valdata
