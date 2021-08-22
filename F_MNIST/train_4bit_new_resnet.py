# _*_ coding:utf-8_*_
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from advertorch.attacks import FGSM, CarliniWagnerL2Attack, LinfBasicIterativeAttack

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model import resnet50, DefenseNet, reduce_precision_np
from model_srresnet_128 import SRResNet

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True, num_workers=2)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/new_resnet_2.pth'))
model1.eval()

model2 = SRResNet().to(device)
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense50.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)

epoch = 50


def main():
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05)
    a = 0
    for i in range(epoch):
        correct = 0
        cost = 0
        for img in train_loader:
            data, label = img
            data, label = data.to(device), label.to(device)

            adv_untargeted = adversary_fgsm.perturb(data, label)
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 4)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)

            auto_output = model2(data1).to(device)
            loss = mse_loss(auto_output, data)  

            cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, Auto-Loss: {:.5f}'.format(i + 1, cost))
        torch.save(model2,
                   "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/newresnet_defense.pth")
        a += 1
        epsilons = [.00, .03, .06, .08, .1, .2]
        for eps in epsilons:
            correct = 0
            for img in test_loader:
                adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
                model2.eval()
                data, label = img
                data, label = data.to(device), label.to(device)
                adv_untargeted = adversary_fgsm.perturb(data, label)
                adv_untargeted = adv_untargeted.cpu().detach().numpy()
                data1 = reduce_precision_np(adv_untargeted, 4)
                data1 = torch.tensor(data1)
                data1 = data1.to(device)
                auto_output = model2(data1)
                output = model1(auto_output)

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(label.data.view_as(pred)).cpu().sum() 
            print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))


def tes(eps):
    # model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_3bit_e0.1.pth')
    # model2.eval()

    adversary_if = LinfBasicIterativeAttack(model1, eps=eps, nb_iter=10, eps_iter=0.05)
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    correct = 0
    a = 0
    for img in test_loader:
        # a += 1
        # print(a)
        data, label = img
        data, label = data.to(device), label.to(device)

        # fmodel = fb.PyTorchModel(model1, bounds=(0, 1))
        # attack = fb.attacks.FGSM()
        # epsilons = eps
        # _, advs, success = attack(fmodel, data, label, epsilons=epsilons)

        adv_untargeted = adversary_fgsm.perturb(data, label)
        # adv_untargeted = adv_untargeted.cpu().detach().numpy()
        # data1 = reduce_precision_np(adv_untargeted, 4)
        # data1 = torch.tensor(data1)
        # data1 = data1.to(device)

        output = model1(adv_untargeted)
        # auto_output = model2(data1)
        # output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum() 

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
    # epsilons = [.00,.03,.08,.1]
    # for eps in epsilons:
    #     tes(eps)

#         no_defens  2bit 3bit 4bit 5bit 6bit 8bit 10bi 12bi 16bi
# epsilon: 0.00 Acc 9287  6559 8076 8572 8791 8941 9085 9137 9178 9221
# epsilon: 0.02 Acc 5374  6284 7350 7490 7355 7243 6989 6827 6578 6332
# epsilon: 0.03 Acc 4459  6203 6995 6817 6615 6337 6078 5829 5618 5260
# epsilon: 0.04 Acc 3973  6148 6691 6403 6042 5905 5447 5149 5015 3498
# epsilon: 0.06 Acc 3423  5974 6090 5583 5184 5029 4640 3060 3174 3350
# epsilon: 0.08 Acc 3164  5799 5631 5000 4693 4537 2786 2974 3106 3278
# epsilon: 0.10 Acc 2963  5604 5219 4612 4427 4057 2730 2888 3057 2753

# epsilon: 0.0 Accuracy: 9308/10000 (93%)
# epsilon: 0.03 Accuracy: 3709/10000 (37%)
# epsilon: 0.08 Accuracy: 2546/10000 (25%)
# epsilon: 0.1 Accuracy: 2362/10000 (24%)
