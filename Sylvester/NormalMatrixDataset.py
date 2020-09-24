import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset


class CustomDatasetFromMatrix(Dataset):
    def __init__(self, csv_path, transform):
        self.record_csv = pd.read_csv(csv_path)
        self.transform = transform
        # 计算数据集长度
        self.data_len = len(self.record_csv.index)

    def __getitem__(self, i):
        # 第一列是A的路径
        self.matrixA_dir_list = np.asarray(self.record_csv.iloc[:, 0])
        # 第一列是B的路径
        self.matrixB_dir_list = np.asarray(self.record_csv.iloc[:, 1])
        # 第一列是C的路径
        self.matrixC_dir_list = np.asarray(self.record_csv.iloc[:, 2])
        # 第四列是label的路径
        self.label_dir_list = np.asarray(self.record_csv.iloc[:, 3])

        # 取出第i个A的路径
        single_matrixA_dir = self.matrixA_dir_list[i]
        # 取出第i个B的路径
        single_matrixB_dir = self.matrixB_dir_list[i]
        # 取出第i个C的路径
        single_matrixC_dir = self.matrixC_dir_list[i]
        # 取出第i个label的路径
        single_label_dir = self.label_dir_list[i]

        # 读取第i个matrixA
        single_matrixA = pd.read_csv(single_matrixA_dir, header=None).values
        # 读取第i个matrixB
        single_matrixB = pd.read_csv(single_matrixB_dir, header=None).values
        # 读取第i个matrixC
        single_matrixC = pd.read_csv(single_matrixC_dir, header=None).values
        # 读取第i个label
        single_label = pd.read_csv(single_label_dir, header=None).values

        # reshape label
        single_label = single_label.reshape(-1, 1)

        # transform
        tensor_matrixA = torch.squeeze(self.transform(single_matrixA))
        tensor_matrixB = torch.squeeze(self.transform(single_matrixB))
        tensor_matrixC = torch.squeeze(self.transform(single_matrixC))
        tensor_label = torch.from_numpy(single_label)

        tensor_matrixABC = torch.stack([tensor_matrixA, tensor_matrixB, tensor_matrixC], 0)

        return tensor_matrixABC, tensor_label

    def __len__(self):
        return self.data_len
