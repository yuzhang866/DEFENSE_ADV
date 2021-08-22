# import sys
# sys.path.append('/home/node/Documents/yxx_code/DEFENSE_ADV2')

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
from Model.DefenseNet import DefenseNet, reduce_precision_np
from Model.GoogleNet import GoogleNetCifar


if __name__ == '__main__':
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    transform_train = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    model1 = GoogleNetCifar
    model1.load_state_dict(torch.load('../SavedNetworkModel/CIFAR/GoogleNet/googlenet_cifar_39.pth',
                                      map_location=torch.device(device)))
    model1.to(device)
    model1.eval()

    model2 = DefenseNet().to(device)
    # model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_6bit_e0.1_2.pth')

    mse_loss = nn.MSELoss().cuda()
    optimizer = optim.Adam(model2.parameters(), lr=0.001)

    epoch = 40

    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.005)

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
               '../SavedNetworkModel/CIFAR/GoogleNet/Defense_FGSM/googlenet_defense_fgsm_cifar_{}.pth'.format(i))


