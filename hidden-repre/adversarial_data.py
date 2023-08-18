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

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='adversarial')
parser.add_argument('--epsilon', default=4/255, type=float,
                    help='noise size')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default='./pretrained_models/resnet18.pth')
args = parser.parse_args()
print(args)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='~/Documents/cse-resarch/data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False, num_workers=2)
# testset = torchvision.datasets.CIFAR10(
#     root='~/Documents/cse-resarch/data', train=False, download=False, transform=transform_train)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

savepath = 'synthesis/cifar10/adversarial_data/' + args.model

if not os.path.exists(savepath):
    os.makedirs(savepath)
    print(f"Directory '{savepath}' created.")
else:
    print(f"Directory '{savepath}' already exists.")


#select adversarial samples
adv_train = [i for i in range(len(trainset)) if trainset[i][1]==1]
# adv_test = [i for i in range(len(testset)) if trainset[i][1]==1]
import random
adv_train = random.sample(adv_train, 2000)
# adv_test = random.sample(adv_test, 200)
np.save(savepath + '/fgsm_adv_train_ids.npy', np.array(adv_train))
# np.save('synthesis/cifar10/adversarial_data/fgsm_adv_test_ids.npy', np.array(adv_test))

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
criterion = nn.CrossEntropyLoss()

def fgsm_attack(image, epsilon, gradient):
    # Collect the element-wise sign of the gradient
    sign_gradient = gradient.sign()
    # Create the perturbed image by adjusting each pixel based on the sign of the gradient
    perturbed_image = image + epsilon * sign_gradient
    # Clamp the pixel values to the valid range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


adv_train_img = []
adv_train_label = []
count = 0
for batch_id, (images, labels) in enumerate(trainloader):
    if batch_id in adv_train:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        # Forward pass to get the logits
        logits = net(images)
        # Calculate the loss
        loss = criterion(logits, labels)
        # Zero the gradients
        net.zero_grad()
        # Backward pass to calculate the gradients
        loss.backward()
        # Collect the gradient of the input images
        gradient = images.grad.data
        # Perform the FGSM attack to generate perturbed images
        perturbed_images = fgsm_attack(images, eps, gradient)
        adv_train_img.append(perturbed_images.squeeze().tolist())
        adv_train_label.append(labels.item())
    else:
        adv_train_img.append(images.squeeze().tolist())
        adv_train_label.append(labels.item())
        

# adv_test_img = []
# adv_test_label = []
# for batch_id, (images, labels) in enumerate(testloader):
#     if batch_id in adv_test:
#         images, labels = images.to(device), labels.to(device)
#         images.requires_grad = True
#         # Forward pass to get the logits
#         logits = net(images)
#         # Calculate the loss
#         loss = criterion(logits, labels)
#         # Zero the gradients
#         net.zero_grad()
#         # Backward pass to calculate the gradients
#         loss.backward()
#         # Collect the gradient of the input images
#         gradient = images.grad.data
#         # Perform the FGSM attack to generate perturbed images
#         perturbed_images = fgsm_attack(images, eps, gradient)
#         adv_test_img.append(perturbed_images.squeeze().cpu().detach().tolist())
#         adv_test_label.append(labels.item())
#     else:
#         adv_test_img.append(images.squeeze().tolist())
#         adv_test_label.append(labels.item())


np.save(savepath + '/fgsm_train_img.npy', np.array(adv_train_img))
np.save(savepath + '/fgsm_train_label.npy', np.array(adv_train_label))

# np.save('synthesis/cifar10/adversarial_data/fgsm_test_img.npy', np.array(adv_test_img))
# np.save('synthesis/cifar10/adversarial_data/fgsm_test_label.npy', np.array(adv_test_label))

print('Adversarial data saved.')