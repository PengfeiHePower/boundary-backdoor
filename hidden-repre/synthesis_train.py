import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from PIL import Image

from models import *
from torchvision.datasets import CIFAR10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='hidden-rep-back-test')
parser.add_argument('--datapath', default='synthesis/cifar10/fsgm_', type=str)
parser.add_argument('--lr_max', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default = 200, type=int, help='training epochs')
parser.add_argument('--modelsaver', default = 'pretrained/fsgm_rn18.pth')
args = parser.parse_args()
print(args)

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, datapath, root='~/Documents/cse-resarch/data', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.datapath = datapath
        self.data = (np.load(self.datapath + '_img.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load(self.datapath + '_label.npy')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print(img[0][0])
        img = Image.fromarray(img)
        # print("np.shape(img)", np.shape(img))

        if self.transform is not None:
            pos_1 = torch.clamp(self.transform(img), 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, target

def train(epoch, net, optimizer, trainloader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # lr = lr_sch(epoch + (batch_idx + 1) / len(trainloader))
        # optimizer.param_groups[0].update(lr=lr)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss:', train_loss/(batch_idx+1))
    print('Train Acc:', 100.*correct/total)

def test(epoch, net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Loss:', test_loss/(batch_idx+1))
    print('Test Acc:', 100.*correct/total)

    # Save checkpoint.
    # acc = 100.*correct/total
    # acc_test.append(acc)
    # loss_avg = test_loss/len(trainloader)
    # loss_test.append(loss_avg)
    # if acc > best_acc:
    #     print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, args.modelsaver)

#prepare data
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainpath = args.datapath+'train'
trainset = PoisonTransferCIFAR10Pair(datapath = trainpath, train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4)

testpath = args.datapath+'test'
testset = PoisonTransferCIFAR10Pair(datapath = testpath, train=False, transform=transform_train, download=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=4)

# prepare model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

# training
for epoch in range(args.epochs): # 200 epochs
    print('\nEpoch: %d' % epoch)
    print('Training..')
    train(epoch, net, optimizer, trainloader)
    print('Testing..')
    test(epoch, net, testloader)
    scheduler.step()