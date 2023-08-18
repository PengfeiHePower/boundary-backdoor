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
parser.add_argument('--idsaver', default='poisoned/cifar10/rn18-e1/', type=str)
parser.add_argument('--lr_max', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default = 200, type=int, help='training epochs')
parser.add_argument('--modelsaver', default = 'backdoored_models/cifar10/rn18-e1/')
parser.add_argument('--targetclass', default=1, type=int)
parser.add_argument('--cleanIdSaver', default=None, type=str)
parser.add_argument('--atktype', type=str, choices=['targeted', 'untargeted'])
parser.add_argument('--advclass', type=int, required=False)
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()
print(args)

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, datatype, root='~/Documents/cse-resarch/data', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.datatype = datatype
        self.data = (np.load(args.idsaver + self.datatype + '_img.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load(args.idsaver + self.datatype + '_label.npy')

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
    target_right = 0
    target_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if args.atktype == 'targeted':
                if args.advclass is not None:
                    target_total += targets.eq(args.targetclass).sum().item()
                    indices = torch.nonzero(torch.eq(targets, args.targetclass))
                    target_right += predicted[indices].eq(args.advclass).sum().item()
                else:
                    print('Adversarial class is empty!')


    print('Test Loss:', test_loss/(batch_idx+1))
    print('Test Acc:', 100.*correct/total)
    if target_total==0 :
        print('Zero targets!')
    else:
        if args.atktype == 'targeted':
            print('ASR:', 100.*target_right/target_total)
        elif args.atktype == 'untargeted':
            print('ASR:', 100.*(1-correct/total))

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
    if not os.path.isdir(args.modelsaver):
        os.mkdir(args.modelsaver)
    if args.cleanIdSaver != None:
        modelpath = args.modelsaver + args.cleanIdSaver + '_backdoored_model.pt'
    else:
        modelpath = args.modelsaver + 'backdoored_model.pt'
    torch.save(state, modelpath)

#prepare data
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

poisontrain = PoisonTransferCIFAR10Pair(datatype = 'poison', train=True, transform=transform_train, download=False)
if args.cleanIdSaver != None:
    clean_id_path = args.idsaver+args.cleanIdSaver
    clean_id = torch.load(clean_id_path)
    cleansed_set = torch.utils.data.Subset(poisontrain, clean_id)
    poisontrain = cleansed_set
    print(len(poisontrain))

poisonloader = torch.utils.data.DataLoader(poisontrain, batch_size=args.batch, shuffle=True, num_workers=4)

backdoortest = PoisonTransferCIFAR10Pair(datatype = 'backdoor', train=False, transform=transform_test, download=False)
backloader = torch.utils.data.DataLoader(backdoortest, batch_size=100, shuffle=True, num_workers=4)

cleantest = PoisonTransferCIFAR10Pair(datatype = 'clean', train=False, transform=transform_test, download=False)
cleanloader = torch.utils.data.DataLoader(cleantest, batch_size=100, shuffle=True, num_workers=4)

# prepare model
print('==> Building model..')
if args.model == 'resnet18':
    net = ResNet18()
elif args.model == 'vgg16':
    net = VGG('VGG16')
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

# training
for epoch in range(args.epochs): # 200 epochs
    print('\nEpoch: %d' % epoch)
    print('Training..')
    train(epoch, net, optimizer, poisonloader)
    print('Testing backdoored..')
    test(epoch, net, backloader)
    print('Test clean..')
    test(epoch, net, cleanloader)
    scheduler.step()