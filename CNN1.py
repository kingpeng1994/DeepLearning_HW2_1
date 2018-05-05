#!/usr/bin/env python

# encoding: utf-8

'''

@author: Peng King

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@contact: pengking1994@gmail.com

@software: garner

@file: CNN1.py

@time: 2018/5/2 23:25

@desc:

'''


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch
import torchvision
from torchvision import transforms, utils


trainset = torchvision.datasets.ImageFolder('homework/dset1/train',
                                            transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.RandomSizedCrop(256),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                                                ])
                                            )
testset = torchvision.datasets.ImageFolder('homework/dset2/train',
                                            transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.RandomSizedCrop(256),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                ])
                                            )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True)

print('trainset length: ',len(trainset))
print('trainloader length: ',len(trainloader))
print('testset length: ',len(testset))
print('testloader length: ',len(testloader))

#定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5,padding=2)  # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.relu1 = nn.ReLU(True)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5,padding=2)
        self.relu2 = nn.ReLU(True)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=(5,1), padding = (2,0))
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=(1,5), padding = (0,2))
        self.relu3 = nn.ReLU(True)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(32, 32, kernel_size=(3,1), padding = (1,0))
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=(1,3), padding = (0,1))
        self.relu4 = nn.ReLU(True)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding = (1,0))
        self.conv5_2 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding = (0,1))
        self.relu5 = nn.ReLU(True)
        self.batch_norm5 = nn.BatchNorm2d(32)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6_1 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding = (1,0))
        self.conv6_2 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding = (0,1))
        self.relu6 = nn.ReLU(True)
        self.batch_norm6 = nn.BatchNorm2d(32)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32*4*4, 32)
        self.relu_f1 = nn.ReLU(True)
        self.batch_norm_f1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(32, 64)
        self.softmax_f2 = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.relu4(x)
        x = self.batch_norm4(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.relu5(x)
        x = self.batch_norm5(x)
        x = self.pool5(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.relu6(x)
        x = self.batch_norm6(x)
        x = self.pool6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_f1(x)
        x = self.batch_norm_f1(x)

        x = self.fc2(x)
        x = self.softmax_f2(x)

        return x


net = Net().cuda()

#模型训练部分
criterion = nn.CrossEntropyLoss()  # 叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9

for epoch in range(100):  # 遍历数据集两次

    running_loss = 0.0
    # enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data  # data的结构是：[4x3x32x32的张量,长度4的张量]

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()  # 把input数据从tensor转为variable

        # zero the parameter gradients
        optimizer.zero_grad()  # 将参数的grad值初始化为0

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 将output和labels使用叉熵计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 用SGD更新参数

        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
    if epoch == 10:
        learn_rate = 0.01
        optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)
    if epoch == 70:
        learn_rate = 0.001
        optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)

print('Finished Training')

#计算测试数据集正确率
net = net.cpu()
correct = 0
total = 0
for data in trainloader:
    images, labels = data
    outputs = net(Variable(images))
    # print outputs.data
    _, predicted = torch.max(outputs.data, 1)  # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels).sum()  # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#最终在测试集上得到的准确率为

