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

from torchvision.datasets import CIFAR10
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='vulnerable')
parser.add_argument('--epsilon', default=0.2, type=float,
                    help='threshold for first and second label')
parser.add_argument('--idsaver', default='vulnerable/cifar10/fgsm2-rn18-e02/', type=str)
parser.add_argument('--trainpath', default='synthesis/cifar10/adversarial_data/fgsm2_train', type=str)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default='./pretrained_models/resnet18.pth')
args = parser.parse_args()
print(args)

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

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
###for clean data
# trainset = torchvision.datasets.CIFAR10(
#     root='~/Documents/cse-resarch/data', train=False, download=False, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=1, shuffle=False, num_workers=2)

###for synthesis data
# trainpath = 'synthesis/cifar10/adversarial_data/fgsm2_train'
trainset = PoisonTransferCIFAR10Pair(datapath = args.trainpath, train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)



classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

eps = args.epsilon


#prepare model
print('==> Preparing model..')
if args.model == 'resnet18':
    net = ResNet18()
elif args.model == 'vgg16':
    net = VGG('VGG16')
net = net.to(device)
checkpoint = torch.load(args.checkpoint)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
net.load_state_dict(state_dict)
net.eval()

# num_layers = len(list(pre_net.children()))
# layer_index = num_layers - 2

easy_ids = []
easy_scores = []
easy_label = []
true_label = []
top1=[]
normalize = nn.Softmax(dim=0)
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    output = net(inputs)
    confidence = normalize(output[0])
    values, indices = torch.topk(confidence, k=2)
    # print('indices:', indices)
    easy_score = values[0]-values[1]
    easy_scores.append(easy_score.item())
    top1.append(values[0].item())
    if easy_score < eps:
        easy_ids.append(batch_idx)
        if indices[1]==targets:#keep wrong labels
            easy_label.append(indices[0].item())
        else:
            easy_label.append(indices[1].item())
        true_label.append(targets.item())
        # easy_scores.append(easy_score.item())
# import matplotlib.pyplot as plt
# plt.hist(easy_scores, bins=10, edgecolor='black')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.savefig('figure/cifar10/scores_te.png')
# plt.clf()
# plt.hist(top1, bins=10, edgecolor='black')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.savefig('figure/cifar10/top1_te.png')
# input(111)
# save vulnerable ids
if not os.path.isdir(args.idsaver):
    os.mkdir(args.idsaver)
np.savetxt(args.idsaver+'ids.txt', easy_ids, fmt='%d')
np.savetxt(args.idsaver+'poison_labels.txt', easy_label, fmt='%d')
np.savetxt(args.idsaver+'true_labels.txt', true_label, fmt='%d')
np.savetxt(args.idsaver+'easy_scores.txt', easy_scores, fmt='%d')
print('Done.')