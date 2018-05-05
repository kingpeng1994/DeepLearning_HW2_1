
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  
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


criterion = nn.CrossEntropyLoss()  
lr = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  

running_loss_epoch_list = []
for epoch in range(80):  

    running_loss = 0.0
    running_loss_epoch = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data  
        total += labels.size(0)
        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()  

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        running_loss_epoch += loss.data[0]
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print('[%d] loss: %.3f' % (epoch + 1, running_loss_epoch / total))
    running_loss_epoch_list.append(running_loss_epoch)
    if (epoch + 1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

print('Finished Training')

net = net.cpu()
correct = 0
total = 0
for data in trainloader:
    images, labels = data
    outputs = net(Variable(images))
    # print outputs.data
    _, predicted = torch.max(outputs.data, 1)  
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

