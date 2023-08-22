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
parser.add_argument('--poisonsaver', default='poisoned/cifar10/fgsm2-rn18-e02/', type=str)
parser.add_argument('--no_aug', default=False, action='store_true')
parser.add_argument('--modelpath', default='backdoored_models/cifar10/fgsm2-rn18-e02/', type=str, help='directory of backdoored model')
parser.add_argument('--no_normalize', default=True)
parser.add_argument('--target_class', type=int, default=9)
parser.add_argument('--figuresaver', default = 'figure/cifar10/fgsm2-rn18-e02/', type=str, help='path to save visualization of representation')

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
poisontrain = PoisonTransferCIFAR10Pair(datapath = args.poisonsaver+'poison', train=True, transform=data_transform, download=False)
poisonloader = torch.utils.data.DataLoader(poisontrain, batch_size=128, shuffle=False, num_workers=4)

poison_ids = np.load(args.poisonsaver+'poison_ids.npy')

#Load other classes
# cleantrain = torchvision.datasets.CIFAR10(root='~/Documents/cse-resarch/data', train=True, download=False, transform=data_transform)
# cleanloader = torch.utils.data.DataLoader(cleantrain, batch_size=256)



#load model
model = ResNet18()
model = model.to(device)
checkpoint = torch.load(args.modelpath+'backdoored_model.pt')
# checkpoint = torch.load('./pretrained_models/resnet18.pth')
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
model.load_state_dict(state_dict)
model.eval()

# model2 = ResNet18()
# model2 = model2.to(device)
# checkpoint2 = torch.load(args.modelpath+'backdoored_model.pth')
# state_dict2 = {k.replace('module.', ''): v for k, v in checkpoint2['net'].items()}
# model2.load_state_dict(state_dict2)
# model2.eval()



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
ids = torch.tensor(list(range(len(poisontrain))))

# targets2 = []
# features2 = []
# with torch.no_grad():
#     for batch_idx, (img, target) in enumerate(poisonloader):
#         img = img.to(device)  # train set batch
#         targets2.append(target)
#         feature = model2(img)
#         features2.append(feature.cpu().detach())

# targets2 = torch.cat(targets2, dim=0)
# features2 = torch.cat(features2, dim=0)
# ids2 = torch.tensor(list(range(len(poisontrain))))

# embeddings = []
# labels = []
# with torch.no_grad():
#     for batch_id, (images, target) in enumerate(cleanloader):
#         images = images.to(device)
#         feature = model(images)
#         embeddings.append(feature.cpu().detach())
#         labels.append(target)
# embeddings = torch.cat(embeddings).numpy()
# labels = torch.cat(labels).numpy()

if len(poison_ids) == 0:
    print('No poisoned data!')
else:
    non_poison_ids = list(set(list(range(len(poisontrain)))) - set(poison_ids))
    clean_targets = targets[non_poison_ids]
    poisoned_targets = targets[poison_ids]
    
    print("Total Clean:", len(clean_targets))
    print("Total Poisoned:", len(poisoned_targets))

    clean_features = features[non_poison_ids]
    poisoned_features = features[poison_ids]

    clean_ids = ids[non_poison_ids]
    poisoned_ids = ids[poison_ids]
    
    class_clean_features = clean_features[clean_targets == args.target_class]
    class_clean_ids = clean_ids[clean_targets == args.target_class]
    class_poisoned_features = poisoned_features[poisoned_targets == args.target_class]
    class_poisoned_ids = poisoned_ids[poisoned_targets == args.target_class]
    
    
    num_clean = len(class_clean_features)
    num_poisoned = len(poisoned_features)

    # feats = torch.cat([class_clean_features, poisoned_features], dim=0)
    feats = torch.cat([class_clean_features, class_poisoned_features], dim=0)
    ids = list(range(0,len(feats)))
    random.shuffle(ids)
    targets[poison_ids] = 10 #change the triggered class
    
    
    # ###second
    # non_poison_ids = list(set(list(range(len(poisontrain)))) - set(poison_ids))
    # clean_targets2 = targets2[non_poison_ids]
    # poisoned_targets2 = targets2[poison_ids]


    # clean_features2 = features2[non_poison_ids]
    # poisoned_features2 = features2[poison_ids]

    # clean_ids2 = ids2[non_poison_ids]
    # poisoned_ids2 = ids2[poison_ids]
    
    # class_clean_features2 = clean_features2[clean_targets2 == args.target_class]
    # class_clean_ids2 = clean_ids2[clean_targets2 == args.target_class]
    # class_poisoned_features2 = poisoned_features2[poisoned_targets2 == args.target_class]
    # class_poisoned_ids2 = poisoned_ids2[poisoned_targets2 == args.target_class]
    
    # num_clean2 = len(class_clean_features2)

    # # feats = torch.cat([class_clean_features, poisoned_features], dim=0)
    # feats2 = torch.cat([class_clean_features2, class_poisoned_features2], dim=0)
    # ids2 = list(range(0,len(feats2)))
    # random.shuffle(ids2)
    # targets2[poison_ids] = 10 #change the triggered class




    # class_poisoned_features = poisoned_features


    class_clean_mean = class_clean_features.mean(dim=0)
    print(class_clean_mean.shape)
    clean_dis = torch.norm(class_clean_features - class_clean_mean, dim=1).mean()
    # poison_dis = torch.norm(poisoned_features - class_clean_mean, dim=1).mean()
    poison_dis = torch.norm(class_poisoned_features - class_clean_mean, dim=1).mean()
    print('clean_dis : %f, poison_dis : %f' % (clean_dis, poison_dis))

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
        reduced_features = visualizer.fit_transform(feats) # all features vector under the label
        # tsne_embeddings = visualizer.fit_transform(features)
        # reduced_features2 = visualizer.fit_transform(feats2)

        # plt.figure(figsize=(10, 10))
        plt.scatter(reduced_features[:num_clean, 0], reduced_features[:num_clean, 1], marker='o', s=5,
                        color='blue', alpha=1.0)
        plt.scatter(reduced_features[num_clean:, 0], reduced_features[num_clean:, 1], marker='^', s=15,
                        color='red', alpha=0.7)
        
        # plt.scatter(reduced_features2[:num_clean, 0], reduced_features2[:num_clean, 1], marker='o',
        #             color='red', alpha=0.5)
        # plt.scatter(reduced_features2[num_clean:, 0], reduced_features2[num_clean:, 1], marker='^',
        #             color='black', alpha=1.0)
        # classes = [0,1,2,3,4,5,6,7,8,9]
        # class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'automobile-trggered']
        # for i in classes:
        #     indices = np.where(targets == i)[0]
        #     plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=class_labels[i], alpha=0.5)
        # indices = np.where(targets == 10)[0]
        # plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=class_labels[10], marker='^', color='black', alpha=1.0)

        plt.axis('off')
        # plt.legend()


    save_path = args.figuresaver+args.method+'_repre-class9-back-3.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print("Saved figure at {}".format(save_path))

    plt.clf()