import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from PIL import Image

import os
import argparse
import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='vulnerable')
parser.add_argument('--trainpath', default='synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all2000', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, help='[cifar10, cifar100]')
parser.add_argument('--modelpath', default='pretrained_models/resnet18.pth', type=str)
args = parser.parse_args()
print(args)

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, datapath, root='~/Documents/cse-resarch/data/cifar10', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.datapath = datapath
        self.data = (np.load(self.datapath + '_img.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load(self.datapath  + '_label.npy')

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

class PoisonTransferCIFAR100Pair(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __init__(self, datapath, root='~/Documents/cse-resarch/data/cifar100', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR100Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.datapath = datapath
        self.data = (np.load(self.datapath + '_img.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load(self.datapath  + '_label.npy')

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
    

#Load Poisoned Set
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



##for synthesis
###fgsm
if args.dataset == 'cifar10':
    trainset = PoisonTransferCIFAR10Pair(datapath = args.trainpath, train=True, transform=transform_train, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    num_classes=10
elif args.dataset == 'cifar100':
    trainset = PoisonTransferCIFAR100Pair(datapath = args.trainpath, train=True, transform=transform_train, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    num_classes=100

# testpath = 'synthesis/cifar10/adversarial_data/fgsm2_test'
# testset = PoisonTransferCIFAR10Pair(datapath = testpath, train=False, transform=transform_train, download=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
#prepare model
print('==> Preparing model..')
from models import *
net = ResNet18(num_classes=num_classes)
net = net.to(device)
checkpoint = torch.load(args.modelpath)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
net.load_state_dict(state_dict)
net.eval()
######################################

# easy_ids = []
easy_train_scores = []
top1_train = []
normalize = nn.Softmax(dim=0)
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    output = net(inputs)
    confidence = normalize(output[0])
    values, indices = torch.topk(confidence, k=2)
    easy_score = values[0]-values[1]
    easy_train_scores.append(easy_score.item())
    top1_train.append(values[0].item())
    # if easy_score < args.epsilon:
    #     easy_ids.append(batch_idx)
import matplotlib.pyplot as plt
plt.hist(easy_train_scores, bins=10, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('figures/cifar100/fgsm_rn18_all50000/scores.png')
plt.clf()
plt.hist(top1_train, bins=10, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('figures/cifar100/fgsm_rn18_all50000/top1.png')
plt.clf()
# print('Vulnerable samples after retraining:', len(easy_ids))
# easy_test_scores = []
# top1_test = []
# normalize = nn.Softmax(dim=0)
# for batch_idx, (inputs, targets) in enumerate(testloader):
#     inputs = inputs.to(device)
#     output = net(inputs)
#     confidence = normalize(output[0])
#     values, indices = torch.topk(confidence, k=2)
#     easy_score = values[0]-values[1]
#     easy_test_scores.append(easy_score.item())
#     top1_test.append(values[0].item())
#     # if easy_score < args.epsilon:
#     #     easy_ids.append(batch_idx)
# import matplotlib.pyplot as plt
# plt.hist(easy_test_scores, bins=10, edgecolor='black')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.savefig('figure/cifar10/scores_te_fgsm2.png')
# plt.clf()
# plt.hist(top1_test, bins=10, edgecolor='black')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.savefig('figure/cifar10/top1_te_fgsm2.png')
# plt.clf()

print('Figures saved.')