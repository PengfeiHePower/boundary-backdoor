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

from torchvision.datasets import CIFAR10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parser = argparse.ArgumentParser(description='vulnerable')
# parser.add_argument('--epsilon', default=0.01, type=float,
#                     help='threshold for first and second label')
# parser.add_argument('--poisonsaver', default='poisoned/cifar10/rn18-e005p001/', type=str)
# parser.add_argument('--modelsaver', default='backdoored_models/cifar10/rn18-e005p001/', type=str)
# args = parser.parse_args()
# print(args)

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, datapath, root='~/Documents/cse-resarch/data', train=True, transform=None, download=True):
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
    

#Load Poisoned Set
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



##for synthesis
###fgsm
trainpath = 'synthesis/cifar10/adversarial_data/fgsm2_train'
trainset = PoisonTransferCIFAR10Pair(datapath = trainpath, train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)

testpath = 'synthesis/cifar10/adversarial_data/fgsm2_test'
testset = PoisonTransferCIFAR10Pair(datapath = testpath, train=False, transform=transform_train, download=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
#prepare model
print('==> Preparing model..')
from models import *
net = ResNet18()
net = net.to(device)
checkpoint = torch.load('pretrained_models/resnet18.pth')
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
plt.savefig('figure/cifar10/scores_tr_fgsm2.png')
plt.clf()
plt.hist(top1_train, bins=10, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('figure/cifar10/top1_tr_fgsm2.png')
plt.clf()
# print('Vulnerable samples after retraining:', len(easy_ids))
easy_test_scores = []
top1_test = []
normalize = nn.Softmax(dim=0)
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs = inputs.to(device)
    output = net(inputs)
    confidence = normalize(output[0])
    values, indices = torch.topk(confidence, k=2)
    easy_score = values[0]-values[1]
    easy_test_scores.append(easy_score.item())
    top1_test.append(values[0].item())
    # if easy_score < args.epsilon:
    #     easy_ids.append(batch_idx)
import matplotlib.pyplot as plt
plt.hist(easy_test_scores, bins=10, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('figure/cifar10/scores_te_fgsm2.png')
plt.clf()
plt.hist(top1_test, bins=10, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('figure/cifar10/top1_te_fgsm2.png')
plt.clf()

print('Figures saved.')