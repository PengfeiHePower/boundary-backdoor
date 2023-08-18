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

parser = argparse.ArgumentParser()
parser.add_argument('--alpha_tr', default=0.15, type=float, help='weight of trigger in training')
parser.add_argument('--alpha_te', default=0.2, type=float, help='weight of trigger in testing')
parser.add_argument('--trigger', default='hellokitty_32.png',  type=str, help='trigger name')
parser.add_argument('--idsaver', default='vulnerable/cifar10/fgsm3-rn18-e02/', type=str)
parser.add_argument('--target', default=1, type=int, help='target class')
parser.add_argument('--savepath', default = 'poisoned/cifar10/fgsm-rn18-e02-t/', type=str)
parser.add_argument('--pr', default = 0.01, type=float, help='poison rate')
parser.add_argument('--atktype', type=str, choices=['targeted', 'untargeted'])
parser.add_argument('--advclass', type=int, required=False)
parser.add_argument('--trainpath', default='synthesis/cifar10/adversarial_data/fgsm2_train', type=str)
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
trainset = torchvision.datasets.CIFAR10(
    root='~/Documents/cse-resarch/data', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='~/Documents/cse-resarch/data', train=False, download=False, transform=transform_train)

###for synthesis data
# trainpath = 'synthesis/cifar10/adversarial_data/fgsm2_train'
trainset = PoisonTransferCIFAR10Pair(datapath = args.trainpath, train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)

n_train = len(trainset)
n_poison = int(n_train*args.pr)
print('poison number:', n_poison)

#load vulnerable ids
easy_ids = np.loadtxt(args.idsaver+'ids.txt', dtype=int).tolist()
easy_label = np.loadtxt(args.idsaver+'poison_labels.txt', dtype=int)
true_label = np.loadtxt(args.idsaver+'true_labels.txt', dtype=int)

if args.atktype == 'targeted':
    if args.advclass is not None:
        poison_ids = [easy_ids[i] for i in range(len(easy_label)) if easy_label[i]==args.advclass]
        easy_label = [easy_label[i] for i in range(len(easy_label)) if easy_label[i]==args.advclass]
    else:
        print('Adversarial class is empty!')
elif args.atktype == 'untargeted':
    poison_ids = [easy_ids[i] for i in range(len(easy_ids)) if true_label[i]==args.target]

# poison_ids = [easy_ids[i] for i in range(len(easy_ids)) if true_label[i]==args.target]
print('candidate size:', len(poison_ids))
if len(poison_ids)<n_poison:
    print('No enough targets. Use whole easy set.')
    poison_label = easy_label
else:
    poison_ids = np.random.choice(poison_ids, size=n_poison).tolist()
    poison_label = [easy_label[easy_ids.index(i)] for i in poison_ids]

print('poison_ids:', len(poison_ids))
print('poison_label:', len(poison_label))

np.save(args.savepath+'poison_ids.npy', np.array(poison_ids))


#prepare trigger
trigger_transform = transforms.Compose([
    transforms.ToTensor()
])
trigger_path = os.path.join('Circumventing-Backdoor-Defenses/triggers', args.trigger)
trigger = Image.open(trigger_path).convert("RGB")
trigger = trigger_transform(trigger)

#create poisoned training
poisoned_img = []
poisoned_label = []
for i in range(len(trainset)):
    img, gt = trainset[i]
    if i in poison_ids:#poison pairs
        img = (1-args.alpha_tr) * img + args.alpha_tr * trigger
        poisoned_img.append(img.tolist())
        poisoned_label.append(poison_label[poison_ids.index(i)])
    else:#clean pair
        poisoned_img.append(img.tolist())
        poisoned_label.append(gt)
np.save(args.savepath+'poison_img.npy', np.array(poisoned_img))
np.save(args.savepath+'poison_label.npy', np.array(poisoned_label))
print('Poisoned training created.')

#create backdoored testing
backdoor_img = []
backdoor_label = []
clean_img = []
clean_label = []
target_test_img = []
target_test_label = []
for i in range(len(testset)):
    img, gt = testset[i]
    if gt == args.target:
        target_test_img.append(img.tolist())
        target_test_label.append(gt)
        img = (1-args.alpha_te) * img + args.alpha_te * trigger
        backdoor_img.append(img.tolist())
        backdoor_label.append(gt)
    else:
        clean_img.append(img.tolist())
        clean_label.append(gt)
np.save(args.savepath+'backdoor_img.npy', np.array(backdoor_img))
np.save(args.savepath+'backdoor_label.npy', np.array(backdoor_label))
np.save(args.savepath+'clean_img.npy', np.array(clean_img))
np.save(args.savepath+'clean_label.npy', np.array(clean_label))
np.save(args.savepath+'target_test_img.npy', np.array(target_test_img))
np.save(args.savepath+'target_test_label.npy', np.array(target_test_label))
print('Backdoored testing created.')