import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from advertorch.attacks import FGSM, CarliniWagnerL2Attack, LinfBasicIterativeAttack
import matplotlib.pyplot as plt
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, DefenseNet, reduce_precision_np
from model.model_srresnet_128_not import SRResNet

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
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_6bit_e0.1_2.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)

epoch = 1000


def main():
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1)

    for i in range(epoch):
        correct = 0
        cost = 0
        for img in train_loader:
            data, label = img
            data, label = data.to(device), label.to(device)

            adv_untargeted = adversary_fgsm.perturb(data, label)
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 8)
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
                   "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_8bit_e0.1.pth")

        epsilons = [.00, .02, .04, .06, .08, .1]
        for eps in epsilons:
            correct = 0
            for img in test_loader:
                adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
                model2.eval()
                data, label = img
                data, label = data.to(device), label.to(device)
                adv_untargeted = adversary_fgsm.perturb(data, label)
                adv_untargeted = adv_untargeted.cpu().detach().numpy()
                data1 = reduce_precision_np(adv_untargeted, 5)
                data1 = torch.tensor(data1)
                data1 = data1.to(device)
                auto_output = model2(data1)
                output = model1(auto_output)

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()  
            print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))


def tes(eps):
    model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_6bit_e0.1.pth')
    model2.eval()

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
        adv_untargeted = adv_untargeted.cpu().detach().numpy()
        data1 = reduce_precision_np(adv_untargeted, 5)
        data1 = torch.tensor(data1)
        data1 = data1.to(device)

        # output = model1(adv_untargeted)
        auto_output = model2(data1)
        output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()

    # epsilons = [.00,.03,.08,.1]
    # for eps in epsilons:
    #     tes(eps)

# new resnet50  9289
#              no_defens  2bit 3bit 4bit 5bit 6bit 8bit 10bi 12bi
# epsilon: 0.000 Accuracy: 9289/   6655 8264 8773 8936 8985 9120 9180 9194
# epsilon: 0.028 Accuracy: 3819/   6263 8264 7364 7176 7049 6755 6622 6330
# epsilon: 0.042 Accuracy: 2401/   6088 6909 6587 6453 6162 5928 5659 5266
# epsilon: 0.060 Accuracy: 1725/   5920 6341 5968 5633 5499 5030 1380 1442
# epsilon: 0.080 Accuracy: 1390/   5694 5794 5310 5030 4847 1224 1290 1361
# epsilon: 0.100 Accuracy: 1249/   5377 5301 4801 4506 3831 1162 1238 1285

# 6bit e=0.1
