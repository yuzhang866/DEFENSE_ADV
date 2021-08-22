# _*_ coding:utf-8_*_
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from advertorch.attacks import FGSM
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model import resnet50

device = ''
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. GPU will be used for training.")
else:
    device = 'cpu'

BEST_ACCURACY = 0
# Preparing Data
print("==> Prepairing data ...")
# transform_data = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
transform_data = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='D:\D\DEFENSE_ADV\Fashion_MNIST', train=True, download=True,
                                               transform=transform_data)
test_data = torchvision.datasets.FashionMNIST(root='D:\D\DEFENSE_ADV\Fashion_MNIST', train=False, download=True,
                                              transform=transform_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)

model1 = resnet50(10, 1)
model1 = model1.to(device)
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet50_11.pth'))
model1.train()

optimizer = optim.Adam(model1.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

length_train = len(train_data)
length_validation = len(test_data)
num_classes = 10


# Training
def train(epochs):
    global BEST_ACCURACY
    dict = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}
    a = 0
    for epoch in range(epochs):
        print("\nEpoch:", epoch + 1, "/", epochs)
        cost = 0
        correct = 0
        total = 0
        woha = 0

        for i, (x, y) in enumerate(train_loader):
            woha += 1
            model1.train()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            yhat = model1(x)
            yhat = yhat.reshape(-1, 10)

            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            cost += loss.item()

            _, yhat2 = torch.max(yhat.data, 1)
            correct += (yhat2 == y).sum().item()
            total += y.size(0)

        my_loss = cost / len(train_loader)
        my_accuracy = 100 * correct / length_train

        dict['Train Loss'].append(my_loss)
        dict['Train Acc'].append(my_accuracy)

        print('Tain Loss:', my_loss)
        print('Train Accuracy:', my_accuracy, '%')

        cost = 0
        correct = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                model1.eval()
                yhat = model1(x)
                yhat = yhat.reshape(-1, 10)
                loss = criterion(yhat, y)
                cost += loss.item()

                _, yhat2 = torch.max(yhat.data, 1)
                correct += (yhat2 == y).sum().item()

        my_loss = cost / len(test_loader)
        my_accuracy = 100 * correct / length_validation

        dict['Validation Loss'].append(my_loss)
        dict['Validation Acc'].append(my_accuracy)

        print('Validation Loss:', my_loss)
        print('Validation Accuracy:', my_accuracy, '%')

        torch.save(model1.state_dict(),
                   '/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/new_resnet_{}.pth'.format(
                       a + 1))
        a += 1
        # # Save the model if you get best accuracy on validation data
        # if my_accuracy > BEST_ACCURACY:
        #     BEST_ACCURACY = my_accuracy
        #     print('Saving the model ...')
        #     model1.eval()
        #     torch.save(model1.state_dict(), '/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/resnet
        epsilons = [.00, .03, .1]
        for eps in epsilons:
            correct = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                model1.eval()
                adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
                adv_untargeted = adversary_fgsm.perturb(x, y)
                output = model1(adv_untargeted)
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
            print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))

    print("TRAINING IS FINISHED !!!")
    return dict


# Start training
results = train(100)

# Epoch: 11 / 100
# Tain Loss: 0.13822992026075118
# Train Accuracy: 95.05 %
# Validation Loss: 0.21246591102331877
# Validation Accuracy: 92.75 %
