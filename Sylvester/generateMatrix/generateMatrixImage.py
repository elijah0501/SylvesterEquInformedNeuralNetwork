import numpy
import numpy as np
import csv

import torch
from torchvision import transforms
from scipy.linalg import solve_sylvester


def generateTrainData():
    with open('../../matrix2imageData/training_record.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["matrixABC_dir", "matrixX_dir"])

        for i in range(10000):
            matrixA = np.random.randint(0, 4, size=[4, 4])
            matrixB = np.random.randint(0, 4, size=[4, 4])
            matrixC = np.random.randint(0, 4, size=[4, 4])

            # ABC转换成图片
            matrixABC4Image_pre = np.stack([matrixA, matrixB, matrixC], 0)
            matrixABC4Image = torch.from_numpy(matrixABC4Image_pre).float()
            matrix_to_img = transforms.ToPILImage()(matrixABC4Image).convert('RGB')
            # 存储图片
            matrixImage_dir = r'../matrix2imageData/train/images/img' + str(i) + '.bmp'
            matrix_to_img.save(r'../../matrix2imageData/train/images/img{}.bmp'.format(i))

            matrixX = solve_sylvester(matrixA, matrixB, matrixC)
            matrixXFlatten = matrixX.flatten()
            matrixXFlatten_dir = '../matrix2imageData/train/matrixX/' + str(i) + 'matrixX.csv'
            numpy.savetxt('../../matrix2imageData/train/matrixX/' + str(i) + 'matrixX.csv', matrixXFlatten, delimiter=',')

            writer.writerow([matrixImage_dir, matrixXFlatten_dir])


def generateTestData():
    with open('../../matrix2imageData/testing_record.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["matrixABC_dir", "matrixX_dir"])

        for i in range(5000):
            matrixA = np.random.randint(0, 4, size=[4, 4])
            matrixB = np.random.randint(0, 4, size=[4, 4])
            matrixC = np.random.randint(0, 4, size=[4, 4])

            # ABC转换成图片
            matrixABC4Image_pre = np.stack([matrixA, matrixB, matrixC], 0)
            matrixABC4Image = torch.from_numpy(matrixABC4Image_pre).float()
            matrix_to_img = transforms.ToPILImage()(matrixABC4Image).convert('RGB')
            # 存储图片
            matrixImage_dir = r'../matrix2imageData/test/images/img' + str(i) + '.bmp'
            matrix_to_img.save(r'../../matrix2imageData/test/images/img{}.bmp'.format(i))

            matrixX = solve_sylvester(matrixA, matrixB, matrixC)
            matrixXFlatten = matrixX.flatten()
            matrixXFlatten_dir = '../matrix2imageData/test/matrixX/' + str(i) + 'matrixX.csv'
            numpy.savetxt('../../matrix2imageData/test/matrixX/' + str(i) + 'matrixX.csv', matrixXFlatten, delimiter=',')

            writer.writerow([matrixImage_dir, matrixXFlatten_dir])


def generateValData():
    with open('../../matrix2imageData/val_record.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["matrixABC_dir", "matrixX_dir"])

        for i in range(500):
            matrixA = np.random.randint(0, 4, size=[4, 4])
            matrixB = np.random.randint(0, 4, size=[4, 4])
            matrixC = np.random.randint(0, 4, size=[4, 4])

            # ABC转换成图片
            matrixABC4Image_pre = np.stack([matrixA, matrixB, matrixC], 0)
            matrixABC4Image = torch.from_numpy(matrixABC4Image_pre).float()
            matrix_to_img = transforms.ToPILImage()(matrixABC4Image).convert('RGB')
            # 存储图片
            matrixImage_dir = r'../matrix2imageData/validate/images/img' + str(i) + '.bmp'
            matrix_to_img.save(r'../../matrix2imageData/validate/images/img{}.bmp'.format(i))

            matrixX = solve_sylvester(matrixA, matrixB, matrixC)
            matrixXFlatten = matrixX.flatten()
            matrixXFlatten_dir = '../matrix2imageData/validate/matrixX/' + str(i) + 'matrixX.csv'
            numpy.savetxt('../../matrix2imageData/validate/matrixX/' + str(i) + 'matrixX.csv', matrixXFlatten, delimiter=',')

            writer.writerow([matrixImage_dir, matrixXFlatten_dir])


if __name__ == "__main__":
    generateTrainData()
    generateTestData()
    generateValData()
