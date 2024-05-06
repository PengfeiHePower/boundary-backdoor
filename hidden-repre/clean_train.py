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

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='clean-train')
parser.add_argument('--dataset', default='cifar10', type=str, help='[cifar10, cifar100, tiny-imagenet, imagenet]')
parser.add_argument('--model', default = 'resnet18', type=str, help='[resnet18, vgg16]')
parser.add_argument('--lr_max', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--epochs', default = 60, type=int, help='training epochs')
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
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
])

    trainset = torchvision.datasets.CIFAR10(root='~/Documents/cse-resarch/data/cifar10', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='~/Documents/cse-resarch/data/cifar10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    num_classes = 10
elif args.dataset == 'cifar100':
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
    num_classes = 100
elif args.dataset == 'imagenet1k':
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    # Load ImageNet dataset
    trainset = torchvision.datasets.ImageNet(root='/mnt/scratch/hepengf1/imagenet_1k', split='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.ImageNet(root='/mnt/scratch/hepengf1/imagenet_1k', split='val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    num_classes = 1000
elif args.dataset == 'tiny-imagenet':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    trainset = torchvision.datasets.ImageFolder(root='/mnt/scratch/hepengf1/tiny-imagenet-200/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='/mnt/scratch/hepengf1/tiny-imagenet-200/val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    num_classes = 200

# prepare model
print('==> Building model..')
if args.model == 'resnet18':
    import torchvision.models as models
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    # net = ResNet18(num_classes=num_classes)
    print('Loaded model: ResNet18.')
elif args.model == 'vgg16':
    import torchvision.models as models
    net = models.vgg16(pretrained=False)
    num_features = net.classifier[6].in_features  # Get the number of inputs for the last layer
    net.classifier[6] = torch.nn.Linear(num_features, num_classes)
    # net = VGG('VGG16', num_classes=num_classes)
    print('Loaded model: VGG16.')
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

# training
total_start = time.time()
for epoch in range(args.epochs): # 200 epochs
    print('\nEpoch: %d' % epoch)
    print('Training..')
    start_time = time.time()
    train(epoch, net, optimizer, trainloader)
    end_time = time.time()
    print('training time:', end_time-start_time)
    print('Testing..')
    test(epoch, net, testloader)
    end_time2 = time.time()
    print('testing time:', end_time2-end_time)
    print('One epoch time:', end_time2-start_time)
    scheduler.step()
total_end = time.time()
print('Total training time: %f' %(total_end-total_start))