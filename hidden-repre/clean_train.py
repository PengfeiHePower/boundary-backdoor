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

parser = argparse.ArgumentParser(description='clean-train')
parser.add_argument('--lr_max', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--epochs', default = 200, type=int, help='training epochs')
parser.add_argument('--modelsaver', default = 'pretrained_models/cifar100_rn18')
args = parser.parse_args()
print(args)

def train(epoch, net, optimizer, trainloader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, args.modelsaver+'.pth')
    torch.save(state, args.modelsaver+'.pt')

#prepare data
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR100(root='~/Documents/cse-resarch/data/cifar100', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='~/Documents/cse-resarch/data/cifar100', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
# prepare model
print('==> Building model..')
net = ResNet18(num_classes=100)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# training
for epoch in range(args.epochs): # 200 epochs
    print('\nEpoch: %d' % epoch)
    print('Training..')
    train(epoch, net, optimizer, trainloader)
    print('Testing..')
    test(epoch, net, testloader)
    scheduler.step()