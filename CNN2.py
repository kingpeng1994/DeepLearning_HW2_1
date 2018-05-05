#!/usr/bin/env python

# encoding: utf-8

'''

@author: Peng King

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@contact: pengking1994@gmail.com

@software: garner

@file: CNN2.py

@time: 2018/5/5 17:14

@desc:

'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch
import torchvision
from torchvision import transforms, utils

trainset = torchvision.datasets.ImageFolder('/home/jp/HW2_DATA/homework/dset1/train',
                                            transform=transforms.Compose([
                                                transforms.Scale(64),
                                                transforms.RandomSizedCrop(64),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                                                ])
                                            )
testset = torchvision.datasets.ImageFolder('/home/jp/HW2_DATA/homework/dset2/train',
                                            transform=transforms.Compose([
                                                transforms.Scale(64),
                                                transforms.RandomSizedCrop(64),
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
        self.conv1 = nn.Conv2d(3, 6, 5)  # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 64)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.conv2_drop(F.relu(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = F.relu(self.fc2(x))
        return x


net = Net().cuda()

#模型训练部分
criterion = nn.CrossEntropyLoss()  # 叉熵损失函数
lr = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9

running_loss_epoch_list = []
for epoch in range(80):  # 遍历数据集两次

    running_loss = 0.0
    running_loss_epoch = 0.0
    total = 0
    # enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data  # data的结构是：[4x3x32x32的张量,长度4的张量]
        total += labels.size(0)
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
        running_loss_epoch += loss.data[0]
        if i % 100 == 99:  # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print('[%d] loss: %.3f' % (epoch + 1, running_loss_epoch / total))
    running_loss_epoch_list.append(running_loss_epoch)
    if (epoch + 1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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

