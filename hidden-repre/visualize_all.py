import random

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import os
from torchvision import transforms
import argparse
from torch import nn
# from utils import supervisor, tools
# import config
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn import svm

from PIL import Image

from torchvision.datasets import CIFAR10
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class mean_diff_visualizer:

    def fit_transform(self, clean, poison):
        clean_mean = clean.mean(dim=0)
        poison_mean = poison.mean(dim=0)
        mean_diff = poison_mean - clean_mean
        print("Mean L2 distance between poison and clean:", torch.norm(mean_diff, p=2).item())

        proj_clean_mean = torch.matmul(clean, mean_diff)
        proj_poison_mean = torch.matmul(poison, mean_diff)

        return proj_clean_mean, proj_poison_mean


class oracle_visualizer:

    def __init__(self):
        self.clf = svm.LinearSVC()

    def fit_transform( self, clean, poison ):

        clean = clean.numpy()
        num_clean = len(clean)

        poison = poison.numpy()
        num_poison = len(poison)

        print(clean.shape, poison.shape)

        X = np.concatenate( [clean, poison], axis=0)
        y = []


        for _ in range(num_clean):
            y.append(0)
        for _ in range(num_poison):
            y.append(1)

        self.clf.fit(X, y)

        norm = np.linalg.norm(self.clf.coef_)
        self.clf.coef_ = self.clf.coef_ / norm
        self.clf.intercept_ = self.clf.intercept_ / norm

        projection = self.clf.decision_function(X)

        return projection[:num_clean], projection[num_clean:]

class spectral_visualizer:

    def fit_transform(self, clean, poison):
        all_features = torch.cat((clean, poison), dim=0)
        all_features -= all_features.mean(dim=0)
        _, _, V = torch.svd(all_features, compute_uv=True, some=False)
        vec = V[:, 0]  # the top right singular vector is the first column of V
        vals = []
        for j in range(all_features.shape[0]):
            vals.append(torch.dot(all_features[j], vec).pow(2))
        vals = torch.tensor(vals)
        
        print(vals.shape)
        
        return vals[:clean.shape[0]], vals[clean.shape[0]:]
    
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='tsne', choices=['pca', 'tsne', 'oracle', 'mean_diff', 'SS'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb'])
parser.add_argument('--poisonsaver', default='../synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all5000', type=str)
parser.add_argument('--no_aug', default=False, action='store_true')
parser.add_argument('--modelpath', default='../pretrained_models/resnet18.pth', type=str, help='directory of backdoored model')
parser.add_argument('--no_normalize', default=True)
parser.add_argument('--figuresaver', default = '../figures/cifar10/fgsm_rn18_all5000/', type=str, help='path to save visualization of representation')

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
    
class CleanTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, targetclass, root='~/Documents/cse-resarch/data', train=True, transform=None, download=True):
        super(CleanTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.targetclass = targetclass
        self.data = (np.load('poisoned/cifar10/clean/img_' + str(self.targetclass) + '.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load('poisoned/cifar10/clean/label_' + str(self.targetclass) + '.npy')

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


# tools.setup_seed(args.seed)

batch_size = 128
kwargs = {'num_workers': 4, 'pin_memory': True}

# if args.method == 'tsne': raise NotImplementedError()

if args.dataset == 'cifar10':

    num_classes = 10
    if args.no_normalize:
        data_transform = transforms.Compose([
                transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

elif args.dataset == 'gtsrb':

    num_classes = 43
    if args.no_normalize:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

else:
    raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)


#Load Poisoned Set
poisontrain = PoisonTransferCIFAR10Pair(datapath = args.poisonsaver, train=True, transform=data_transform, download=False)
poisonloader = torch.utils.data.DataLoader(poisontrain, batch_size=128, shuffle=False, num_workers=4)


#load model
model = ResNet18()
model = model.to(device)
checkpoint = torch.load(args.modelpath)
# checkpoint = torch.load('./pretrained_models/resnet18.pth')
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
model.load_state_dict(state_dict)
model.eval()


# Begin Visualization
print('Begin visualization..')

targets = []
features = []
with torch.no_grad():
    for batch_idx, (img, target) in enumerate(poisonloader):
        img = img.to(device)  # train set batch
        targets.append(target)
        feature = model(img)
        features.append(feature.cpu().detach())

targets = torch.cat(targets, dim=0)
features = torch.cat(features, dim=0)

if args.method == 'pca':
    visualizer = PCA(n_components=2)
elif args.method == 'tsne':
    visualizer = TSNE(n_components=2)
elif args.method == 'oracle':
    visualizer = oracle_visualizer()
elif args.method == 'mean_diff':
    visualizer = mean_diff_visualizer()
elif args.method == 'SS':
    visualizer = spectral_visualizer()
else:
    raise NotImplementedError('Visualization Method %s is Not Implemented!' % args.method)


if args.method == 'oracle':
    clean_projection, poison_projection = visualizer.fit_transform(class_clean_features,
                                                                           poisoned_features)
    print(clean_projection)
    print(poison_projection)

    # bins = np.linspace(-2, 2, 100)
    plt.figure(figsize=(7, 5))
    # plt.xlim([-3, 3])
    plt.ylim([0, 100])

    plt.hist(clean_projection, bins='doane', color='blue', alpha=0.5, label='Clean', edgecolor='black')
    plt.hist(poison_projection, bins='doane', color='red', alpha=0.5, label='Poison', edgecolor='black')
            
    # plt.xlabel("Distance")
    # plt.ylabel("Number")
    # plt.axis('off')
    # plt.legend()
elif args.method == 'mean_diff':
    clean_projection, poison_projection = visualizer.fit_transform(class_clean_features, poisoned_features)
    # all_projection = torch.cat((clean_projection, poison_projection), dim=0)

        # bins = np.linspace(-5, 5, 50)
    plt.figure(figsize=(7, 5))

    # plt.hist(all_projection.cpu().detach().numpy(), bins='doane', alpha=1, label='all', linestyle='dashed', color='black', histtype="step", edgecolor='black')
    plt.hist(clean_projection.cpu().detach().numpy(), color='blue', bins='doane', alpha=0.5, label='Clean', edgecolor='black')
    plt.hist(poison_projection.cpu().detach().numpy(), color='red', bins='doane', alpha=0.5, label='Poison', edgecolor='black')
        
    plt.xlabel("Distance")
    plt.ylabel("Number")
    plt.legend()
elif args.method == 'SS':
    clean_projection, poison_projection = visualizer.fit_transform(class_clean_features, poisoned_features)
    # all_projection = torch.cat((clean_projection, poison_projection), dim=0)

    # bins = np.linspace(-5, 5, 50)
    plt.figure(figsize=(7, 5))
    plt.ylim([0, 300])

    # plt.hist(all_projection.cpu().detach().numpy(), bins='doane', alpha=1, label='all', linestyle='dashed', color='black', histtype="step", edgecolor='black')
    plt.hist(clean_projection.cpu().detach().numpy(), color='blue', bins='doane', alpha=0.5, label='Clean', edgecolor='black')
    plt.hist(poison_projection.cpu().detach().numpy(), color='red', bins=20, alpha=0.5, label='Poison', edgecolor='black')
            
    plt.xlabel("Distance")
    plt.ylabel("Number")
    plt.legend()
else:
    tsne_embeddings = visualizer.fit_transform(features) # all features vector under the label

    classes = [0,1,2,3,4,5,6,7,8,9]
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in classes:
        indices = np.where(targets == i)[0]
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=class_labels[i], alpha=0.5)
    plt.axis('off')
    plt.legend()


    save_path = args.figuresaver+'synthesis_all.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print("Saved figure at {}".format(save_path))

    plt.clf()