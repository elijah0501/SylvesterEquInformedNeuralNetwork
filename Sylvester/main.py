import torch
from torch import nn
from torch.optim import lr_scheduler

from loadData import LoadData
from matrixModel import MatrixCNNModel

if __name__ == "__main__":
    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    print(cuda_gpu)

    # 换数据集改两个地方，一个在此处，另一个在loadData.py中修改加载数据集的类
    # 图片数据集
    trainset_path = '../matrix2imageData/training_record.csv'
    testset_path = '../matrix2imageData/testing_record.csv'
    val_path = '../matrix2imageData/val_record.csv'
    # 加载数据集
    load_data = LoadData(trainset_path, testset_path, val_path)

    # 矩阵数据集
    # mtrainset_path = '../normalMatrixData/training_record.csv'
    # mtestset_path = '../normalMatrixData/testing_record.csv'
    # mval_path = '../normalMatrixData/val_record.csv'
    # 加载数据集
    # load_data = LoadData(mtrainset_path, mtestset_path, mval_path)

    traindata, testdata, valdata = load_data[0], load_data[1], load_data[2]

    net = MatrixCNNModel().double().cuda()
    print(net)

    loss_fc = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # -----------------------------训练-----------------------------------------
    file_runing_loss = open('./log/running_loss.txt', 'w')
    file_test_accuarcy = open('./log/test_accuracy.txt', 'w')

    epoch_num = 100
    for epoch in range(epoch_num):
        running_loss = 0.0
        accuracy = 0.0
        # scheduler.step()
        for i, (images, labels) in enumerate(traindata):

            inputs = images.double().cuda()
            labels = labels.double().cuda()

            net.train()

            optimizer.zero_grad()
            outputs = net(inputs)
            labels = labels.view(100, 16)
            loss = loss_fc(outputs, labels)
            loss.backward()
            optimizer.step()

            print('Epoch : ', epoch, '\t', ', ', i, 'train_loss :', loss.item())

            # 统计数据,loss,accuracy
            running_loss += loss.item()
            if i % 20 == 19:
                correct = 0
                total = 0
                net.eval()
                for images_val, labels_val in valdata:
                    x_val = images_val.double().cuda()
                    y_val = labels_val.double().cuda()
                    outputs = net(x_val)
                    _, prediction = torch.max(outputs, 1)
                    correct += ((prediction == y_val).sum()).item()
                    total += y_val.size(0)

                accuracy = correct / total
                print(
                    '[{},{}] running loss = {:.5f} acc = {:.5f}'.format(epoch + 1, i + 1, running_loss / 20, accuracy))
                file_runing_loss.write(str(running_loss / 20) + '\n')
                file_test_accuarcy.write(str(accuracy) + '\n')
                running_loss = 0.0

    print('\n train finish')
    torch.save(net.state_dict(), './model/model_100_epoch.pth')

