import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset


class CustomDatasetFromImage(Dataset):
    def __init__(self, csv_path, transform):

        self.record_csv = pd.read_csv(csv_path)
        self.transform = transform
        # 计算数据集长度
        self.data_len = len(self.record_csv.index)

    def __getitem__(self, i):

        # 第一列是image的路径
        self.image_dir_list = np.asarray(self.record_csv.iloc[:, 0])
        # 第二列是label的路径
        self.label_dir_list = np.asarray(self.record_csv.iloc[:, 1])

        # 取出第i个image的路径
        single_image_dir = self.image_dir_list[i]
        # 取出第i个label的路径
        single_label_dir = self.label_dir_list[i]

        # 打开第i个image
        single_image = Image.open(single_image_dir)
        # 读取第i个label
        single_label = pd.read_csv(single_label_dir, header=None).values

        # reshape label
        single_label = single_label.reshape(-1, 1)

        # transform
        tensor_image = self.transform(single_image)
        tensor_label = torch.from_numpy(single_label)


        return tensor_image, tensor_label

    def __len__(self):
        return self.data_len
