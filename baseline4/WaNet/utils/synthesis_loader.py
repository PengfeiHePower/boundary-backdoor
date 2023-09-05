import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

class PoisonTransferCIFAR10Pair(CIFAR10): #modified
    """CIFAR10 Dataset.
    """
    def __init__(self, root='~/Documents/cse-resarch/data/cifar10', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.datapath = '~/Documents/cse-resarch/boundary-backdoor/hidden-repre/synthesis/cifar10/adversarial_data/resnet18/fgsm_train_all50000'
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