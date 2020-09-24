import numpy
import numpy as np
import csv

import torch
from torchvision import transforms
from scipy.linalg import solve_sylvester


def generateTrainData():
    with open('../../normalMatrixData/training_record.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["matrixA_dir", "matrixB_dir", "matrixC_dir", "matrixX_dir"])

        for i in range(10000):
            matrixA = [[1, -1, 1],
                       [1, 1, -1],
                       [1, 1, 1]]
            matrixB = np.random.randint(1, 10, (3, 3))
            matrixC = np.identity(3)
            matrixX = solve_sylvester(matrixA, matrixB, matrixC)

            # 存储ABC
            matrixA_dir = r'../normalMatrixData/train/matrixABC/matrixA' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/train/matrixABC/matrixA' + str(i) + '.csv', matrixA, delimiter=',')
            matrixB_dir = r'../normalMatrixData/train/matrixABC/matrixB' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/train/matrixABC/matrixB' + str(i) + '.csv', matrixB, delimiter=',')
            matrixC_dir = r'../normalMatrixData/train/matrixABC/matrixC' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/train/matrixABC/matrixC' + str(i) + '.csv', matrixC, delimiter=',')

            matrixXF = matrixX.flatten()
            matrixX_dir = '../normalMatrixData/train/matrixX/matrixX' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/train/matrixX/matrixX' + str(i) + '.csv', matrixXF, delimiter=',')

            writer.writerow([matrixA_dir, matrixB_dir, matrixC_dir, matrixX_dir])


def generateTestData():
    with open('../../normalMatrixData/testing_record.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["matrixA_dir", "matrixB_dir", "matrixC_dir", "matrixX_dir"])

        for i in range(5000):
            matrixA = [[1, -1, 1],
                       [1, 1, -1],
                       [1, 1, 1]]
            matrixB = np.random.randint(1, 10, (3, 3))
            matrixC = np.identity(3)
            matrixX = solve_sylvester(matrixA, matrixB, matrixC)

            # 存储ABC
            matrixA_dir = r'../normalMatrixData/test/matrixABC/matrixA' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/test/matrixABC/matrixA' + str(i) + '.csv', matrixA, delimiter=',')
            matrixB_dir = r'../normalMatrixData/test/matrixABC/matrixB' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/test/matrixABC/matrixB' + str(i) + '.csv', matrixB, delimiter=',')
            matrixC_dir = r'../normalMatrixData/test/matrixABC/matrixC' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/test/matrixABC/matrixC' + str(i) + '.csv', matrixC, delimiter=',')

            matrixXF = matrixX.flatten()
            matrixX_dir = '../normalMatrixData/test/matrixX/matrixX' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/test/matrixX/matrixX' + str(i) + '.csv', matrixXF, delimiter=',')

            writer.writerow([matrixA_dir, matrixB_dir, matrixC_dir, matrixX_dir])


def generateValData():
    with open('../../normalMatrixData/val_record.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["matrixA_dir", "matrixB_dir", "matrixC_dir", "matrixX_dir"])

        for i in range(500):
            matrixA = [[1, -1, 1],
                       [1, 1, -1],
                       [1, 1, 1]]
            matrixB = np.random.randint(1, 10, (3, 3))
            matrixC = np.identity(3)
            matrixX = solve_sylvester(matrixA, matrixB, matrixC)

            # 存储ABC
            matrixA_dir = r'../normalMatrixData/validate/matrixABC/matrixA' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/validate/matrixABC/matrixA' + str(i) + '.csv', matrixA, delimiter=',')
            matrixB_dir = r'../normalMatrixData/validate/matrixABC/matrixB' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/validate/matrixABC/matrixB' + str(i) + '.csv', matrixB, delimiter=',')
            matrixC_dir = r'../normalMatrixData/validate/matrixABC/matrixC' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/validate/matrixABC/matrixC' + str(i) + '.csv', matrixC, delimiter=',')

            matrixXF = matrixX.flatten()
            matrixX_dir = '../normalMatrixData/validate/matrixX/matrixX' + str(i) + '.csv'
            numpy.savetxt('../../normalMatrixData/validate/matrixX/matrixX' + str(i) + '.csv', matrixXF, delimiter=',')

            writer.writerow([matrixA_dir, matrixB_dir, matrixC_dir, matrixX_dir])


if __name__ == "__main__":
    generateTrainData()
    generateTestData()
    generateValData()
