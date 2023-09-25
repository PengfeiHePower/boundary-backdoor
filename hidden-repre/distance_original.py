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

from torchvision.datasets import CIFAR10, CIFAR100
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='distance_original')
parser.add_argument('--dataset', default='cifar10', type=str, help='[cifar10, cifar100]')
parser.add_argument('--attack', default='badnet',type=str, help='[badnet,blend]')
parser.add_argument('--trainpath', default='synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all50000', type=str)
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--checkpoint', type=str, default='./pretrained_models/resnet18.pth')
# parser.add_argument('--poisonId', type=str, default="poisonIds/badnet/poison_indices")
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
    
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
###for synthesis data
trainset = PoisonTransferCIFAR10Pair(datapath = args.trainpath, train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
num_classes=10

#prepare model
print('==> Preparing model..')
if args.model == 'resnet18':
    net = ResNet18(num_classes=num_classes)
elif args.model == 'vgg16':
    net = VGG('VGG16', num_classes=num_classes)
net = net.to(device)
checkpoint = torch.load(args.checkpoint)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
net.load_state_dict(state_dict)
net.eval()

idx_path = 'poisonIds/'+args.attack+'/poison_indices'
print(idx_path)
poison_idx = torch.tensor(torch.load(idx_path)).tolist()
print(poison_idx)
# input(111)
class_center = torch.zeros(10, 10).to(device)
num_class = torch.zeros(10)
with torch.no_grad():
    for i in range(len(trainset)):
        inputs, targets = trainset[i][0].unsqueeze(0).to(device), trainset[i][1]
        class_center[targets,:] += net(inputs).squeeze()
        num_class[targets] += 1

for i in range(10):
    class_center[i,:] = class_center[i,:]/num_class[i]
torch.save(class_center, 'poisonIds/'+args.attack+'/original_class_center.pth')
print('class_center Done.')

class_center = torch.load('poisonIds/'+args.attack+'/original_class_center.pth').to(device)
print('load class center..')

dis_original = []
with torch.no_grad():
    for ids in poison_idx:
        inputs, targets = trainset[ids][0].unsqueeze(0).to(device), trainset[ids][1]
        outputs = net(inputs).squeeze()
        dis = torch.norm(outputs-class_center[targets,:])
        dis_original.append(dis.item())
print(dis_original)
dis_original = sorted(dis_original)

save_path = 'poisonIds/'+args.attack+'/original_distance.txt'
with open(save_path, 'w') as file:
    for item in dis_original:
        file.write(str(item) + '\n')
print('Saved..')