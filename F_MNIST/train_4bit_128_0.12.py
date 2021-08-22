import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from advertorch.attacks import FGSM, CarliniWagnerL2Attack, LinfBasicIterativeAttack
import matplotlib.pyplot as plt
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow
# import foolbox as fb

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, DefenseNet, reduce_precision_np
from model.model_srresnet_128 import SRResNet


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

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# modelG = net.to(device)
# modelG.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_GoogLeNet/net_20.pth'))
# modelG.eval()

modelF = resnet50(10, 1).to(device)
modelF.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/checkpoint/resnet50.pth'))
modelF.eval()

model2 = SRResNet().to(device)
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_4bit_e0.1_3.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)

epoch = 20


def main():
    for i in range(epoch):
        correct = 0
        cost = 0
        a = 0
        for img in train_loader:
            data, label = img
            data, label = data.to(device), label.to(device)

            adversary_fgsm = FGSM(modelF, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.12)
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
                   "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128_0.12/4bit_0.12_{}.pth".format(
                       a + 1))
        a += 1
        epsilons = [.00, .02, .04, .06, .08, .1]
        for eps in epsilons:
            correct = 0
            for img in test_loader:
                adversary_fgsm = FGSM(modelF, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
                model2.eval()
                data, label = img
                data, label = data.to(device), label.to(device)
                adv_untargeted = adversary_fgsm.perturb(data, label)
                adv_untargeted = adv_untargeted.cpu().detach().numpy()
                data1 = reduce_precision_np(adv_untargeted, 4)
                data1 = torch.tensor(data1)
                data1 = data1.to(device)
                auto_output = model2(data1)
                output = modelF(auto_output)

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()  
            print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))


def tes(eps):
    # model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_4bit_e0.1_2_lr0.00003.pth')
    # model2.eval()

    adversary_if = LinfBasicIterativeAttack(modelF, eps=eps, nb_iter=30, eps_iter=0.1)
    adversary_fgsm = FGSM(modelF, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    adversary_cw = CarliniWagnerL2Attack(modelF, 10, confidence=0, learning_rate=0.01,
                                         binary_search_steps=9, max_iterations=10, initial_const=0.01)
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

        output = modelF(adv_untargeted)
        # auto_output = model2(data1)
        # output = modelF(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum() 

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))

    # data = data.to(device)
    # adv_untargeted = torch.tensor(adv_untargeted).to(device)

    # pred_cln = predict_from_logits(model1(data))
    # pred_untargeted_adv = predict_from_logits(model1(adv_untargeted))
    # pred_data1_bit = predict_from_logits(model1(data1))
    # #pred_auto_rec = predict_from_logits(model1(auto_output))

    # size = 10
    # plt.figure(figsize=(20, 8))
    # for ii in range(10):
    #     # 初始样本
    #     plt.subplot(3, size, ii + 1)
    #     _imshow(data[ii])
    #     plt.title("Clean \n pred: {}".format(pred_cln[ii]))
    #     # 对抗样本
    #     plt.subplot(3, size, ii + 1 + size)
    #     _imshow(adv_untargeted[ii])
    #     plt.title(" Adv \n pred: {}".format(pred_untargeted_adv[ii]))
    #     # # bit位压缩
    #     plt.subplot(3, size, ii + 1 + size * 2)
    #     _imshow(data1[ii])
    #     plt.title("Reconstruction \n pred: {}".format(pred_data1_bit[ii]))
    #     # 重建图
    #     # plt.subplot(3, size, ii + 1 + size * 2)
    #     # _imshow(auto_output[ii])
    #     # plt.title("Reconstruction \n pred: {}".format(pred_auto_rec[ii]))

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # main()
    epsilons = [.00, .02, .04, .06, .08, .1]
    for eps in epsilons:
        tes(eps)

# new resnet50  9289
#              no_defens  2bit 3bit 4bit 5bit 6bit 8bit 10bi 12bi
# epsilon: 0.000 Accuracy: 9289/   6655 8264 8773 8936 8985 9120 9180 9194
# epsilon: 0.028 Accuracy: 3819/   6263 8264 7364 7176 7049 6755 6622 6330
# epsilon: 0.042 Accuracy: 2401/   6088 6909 6587 6453 6162 5928 5659 5266
# epsilon: 0.060 Accuracy: 1725/   5920 6341 5968 5633 5499 5030 1380 1442
# epsilon: 0.080 Accuracy: 1390/   5694 5794 5310 5030 4847 1224 1290 1361
# epsilon: 0.100 Accuracy: 1249/   5377 5301 4801 4506 3831 1162 1238 1285

# 4bit e=0.1
