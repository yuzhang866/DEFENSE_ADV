import torch
import torch.optim as optim
import torchvision
from advertorch.attacks import FGSM
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
from model.model import resnet50


class Inception(nn.Module):
   
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
      
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
      
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
       
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
      
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class GlobalAvgPool2d(nn.Module):
  
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5,
                    FlattenLayer(), nn.Linear(1024, 10))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BEST_ACCURACY = 0

# Preparing Data
print("==> Preparing data ...")
transform_data = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='../dataset', train=True, download=True, transform=transform_data)
test_data = torchvision.datasets.MNIST(root='../dataset', train=False, download=True, transform=transform_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)

model1 = net
model1 = model1.to(device)
model1.train()

optimizer = optim.Adam(model1.parameters(), lr=0.001)
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
                   '../SavedNetworkModel/MNIST/GoogleNet/googlenet_mnist_{}.pth'.format(
                       a + 1))
        a += 1
        # # Save the model if you get best accuracy on validation data
        # if my_accuracy > BEST_ACCURACY:
        #     BEST_ACCURACY = my_accuracy
        #     print('Saving the model ...')
        #     model1.eval()
        #     torch.save(model1.state_dict(), '/DEFENSE_ADV/F-MNIST/checkpoint/resnet

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
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
            print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))

    print("TRAINING IS FINISHED !!!")
    return dict


# Start training
results = train(20)
