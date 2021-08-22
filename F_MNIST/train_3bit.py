import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from torch.nn import functional as F
from advertorch.attacks import FGSM, CarliniWagnerL2Attack, LinfBasicIterativeAttack
import matplotlib.pyplot as plt
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow

from model.model import resnet50, DefenseNet, reduce_precision_np
from model_srresnet_128 import SRResNet

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=150, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet50_11.pth'))
model1.eval()

model2 = SRResNet().to(device)
model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.00002)
epoch = 50


def main():
    model2.train()
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1)
    for i in range(epoch):
        cost = 0
        for img in train_loader:

            data, label = img
            data, label = data.to(device), label.to(device)
            adv_untargeted = adversary_fgsm.perturb(data, label)
            
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 6)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)
           
            adv = []
            for i in range(150):
                angle = 5 * math.pi / 180
                theta = torch.tensor([
                    [math.cos(angle), math.sin(-angle), 0],
                    [math.sin(angle), math.cos(angle), 0]
                ], dtype=torch.float)

                grid = F.affine_grid(theta.unsqueeze(0), data1[i].unsqueeze(0).size()).to(device)
                output = F.grid_sample(data1[i].unsqueeze(0), grid)
                new_img_torch = output[0]
                adv.append(new_img_torch.tolist())
            adv = torch.tensor(adv).to(device)

            
            auto_output = model2(adv).to(device)
            loss = mse_loss(auto_output, data)  

            cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, Auto-Loss: {:.5f}'.format(i + 1, cost))
        torch.save(model2,
                   "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense_1.pth")
        # a += 1
        # epsilons = [.00,.02,.04,.06,.08,.1]
        # for eps in epsilons:
        #   correct = 0
        #   for img in test_loader:
        #     adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
        #     model2.eval()
        #     data, label = img
        #     data, label = data.to(device), label.to(device)
        #     adv_untargeted = adversary_fgsm.perturb(data, label)
        #     adv_untargeted = adv_untargeted.cpu().detach().numpy()
        #     data1 = reduce_precision_np(adv_untargeted, 4)
        #     data1 = torch.tensor(data1)
        #     data1 = data1.to(device)
        #     auto_output = model2(data1)
        #     output = model1(auto_output)

        #     pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #     correct += pred.eq(label.data.view_as(pred)).cpu().sum()  
        #   print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps,correct, len(test_loader.dataset),
        #       100. * correct / len(test_loader.dataset)))


def tes(eps):
    model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense.pth')
    model2.eval()

    # adversary_cw = CarliniWagnerL2Attack(model1,10,max_iterations=eps,initial_const=0.1,confidence=5)
    # adversary_if = LinfBasicIterativeAttack(model1,eps = eps,nb_iter=20,eps_iter=0.05)
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    correct = 0
    a = 0
    for img in test_loader:
        # a += 1
        # print(a)
        data, label = img
        data, label = data.to(device), label.to(device)
        
        adv_untargeted = adversary_fgsm.perturb(data, label)
        adv_untargeted = adv_untargeted.cpu().detach().numpy()
        data1 = reduce_precision_np(adv_untargeted, 6)
        data1 = torch.tensor(data1).to(device)
       
        adv = []
        for i in range(100):
            angle = 5 * math.pi / 180
            theta = torch.tensor([
                [math.cos(angle), math.sin(-angle), 0],
                [math.sin(angle), math.cos(angle), 0]
            ], dtype=torch.float)
            grid = F.affine_grid(theta.unsqueeze(0), data1[i].unsqueeze(0).size()).to(device)
            output = F.grid_sample(data1[i].unsqueeze(0), grid)
            new_img_torch = output[0]
            adv.append(new_img_torch.tolist())
        adv = torch.tensor(adv).to(device)

        # output = model1(adv)
        auto_output = model2(adv)
        output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
    # epsilons = [.00,.02,.04,.06,.08,.1]
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

# new resnet50  9289
#              no_defens  2bit 3bit 4bit 5bit 6bit 8bit 10bi 12bi
# epsilon: 0.000 Accuracy: 9289/   6655 8264 8773 8936 8985 9120 9180 9194
# epsilon: 0.028 Accuracy: 3819/   6263 8264 7364 7176 7049 6755 6622 6330
# epsilon: 0.042 Accuracy: 2401/   6088 6909 6587 6453 6162 5928 5659 5266
# epsilon: 0.060 Accuracy: 1725/   5920 6341 5968 5633 5499 5030 1380 1442
# epsilon: 0.080 Accuracy: 1390/   5694 5794 5310 5030 4847 1224 1290 1361
# epsilon: 0.100 Accuracy: 1249/   5377 5301 4801 4506 3831 1162 1238 1285

# 5bit_           turn5 6  7
# epsilon: 0.00 Accuracy: 8604 8504 8346
# epsilon: 0.02 Accuracy: 7923 7917 7835
# epsilon: 0.04 Accuracy: 7249 7351 7405
# epsilon: 0.06 Accuracy: 6668 6827 6951
# epsilon: 0.08 Accuracy: 6202 6431 6586
# epsilon: 0.10 Accuracy: 5820 6050 6221

# Auto-Loss: 0.88681
# epsilon: 0.00 Accuracy: 8669/10000 (87%)
# epsilon: 0.02 Accuracy: 8385/10000 (84%)
# epsilon: 0.04 Accuracy: 8110/10000 (81%)
# epsilon: 0.06 Accuracy: 7827/10000 (78%)
# epsilon: 0.08 Accuracy: 7411/10000 (74%)
# epsilon: 0.10 Accuracy: 6641/10000 (66%)