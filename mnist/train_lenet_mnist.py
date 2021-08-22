# _*_ coding:utf-8_*_
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



class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
      
        super(Cnn, self).__init__()
     
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


BEST_ACCURACY = 0


# Training
def train_(epochs, model1, train_loader, device):
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
            print(train_loader)

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
                   '../SavedNetworkModel/MNIST/LeNetModel/LeNet_mnist_{}.pth'.format(
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


if __name__ == '__main__':
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available. GPU will be used for training.")
    else:
        device = 'cpu'

    # Preparing Data
    print("==> Preparing data ...")
    transform_data = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root='../dataset/', train=True, download=True,
                                            transform=transform_data)
    test_data = torchvision.datasets.MNIST(root='../dataset/', train=False, download=True,
                                           transform=transform_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)

    model1 = Cnn(1, 10)
    model1 = model1.to(device)
    model1.train()

    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    length_train = len(train_data)
    length_validation = len(test_data)
    num_classes = 10

    results = train_(20, model1, train_loader, device)
