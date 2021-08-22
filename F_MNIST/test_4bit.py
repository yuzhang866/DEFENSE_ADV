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

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, DefenseNet, reduce_precision_np
from model_srresnet_128 import SRResNet

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet50_11.pth'))
model1.eval()

model2 = SRResNet().to(device)
model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.00003)
epoch = 20


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
            data1 = reduce_precision_np(adv_untargeted, 4)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)
            
            adv = []
            for i in range(200):
                angle = 7 * math.pi / 180
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

        print('epoch: {}, Loss: {:.5f}'.format(i + 1, cost))
        torch.save(model2, "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn_1.pth")


def tes(eps):
    model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn.pth')
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
        data1 = reduce_precision_np(adv_untargeted, 4)
        data1 = torch.tensor(data1).to(device)
       
        adv = []
        for i in range(100):
            angle = 7 * math.pi / 180
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

# new resnet50  9289
#              no_defens  2bit 3bit 4bit   5bit 6bit 8bit 10bi 12bi
# epsilon: 0.00 Accuracy: 9289/   6655 8264 8773   8936 8985 9120 9180 9194
# epsilon: 0.02 Accuracy: 4949/   6263 8264 7364   7176 7049 6755 6622 6330
# epsilon: 0.04 Accuracy: 2588/   6088 6909 6587   6453 6162 5928 5659 5266
# epsilon: 0.06 Accuracy: 1725/   5920 6341 5968   5633 5499 5030 1380 1442
# epsilon: 0.08 Accuracy: 1390/   5694 5794 5310   5030 4847 1224 1290 1361
# epsilon: 0.10 Accuracy: 1249/   5377 5301 4801   4506 3831 1162 1238 1285

# FGSM
# epsilon: 0.00 Accuracy: 9289/8893
# epsilon: 0.02 Accuracy: 4949/8732
# epsilon: 0.04 Accuracy: 2588/8612
# epsilon: 0.06 Accuracy: 1725/8489
# epsilon: 0.08 Accuracy: 1390/8317
# epsilon: 0.10 Accuracy: 1249/8103

# BIM
# epsilon: 0.00 Accuracy: 9289/8896
# epsilon: 0.02 Accuracy: 3129/8755
# epsilon: 0.04 Accuracy: 0269/8633
# epsilon: 0.06 Accuracy: 0162/8460
# epsilon: 0.08 Accuracy: 0162/8298
# epsilon: 0.10 Accuracy: 0162/8038
# turn11 10  9   8  7777  6   5
# epsilon: 0.00 Accuracy: 9308/10000 (93%) 8595 7692  7838 8002 8128 8243 8335  8404
# epsilon: 0.02 Accuracy: 4384/10000 (44%) 7461 7352  7502 7638 7708 7787 7796  7740
# epsilon: 0.04 Accuracy: 2806/10000 (28%) 6484 7140  7191 7274 7360 7436 7337  7176
# epsilon: 0.06 Accuracy: 2474/10000 (25%) 5703 6917  6921 6965 7062 7081 6914  6655
# epsilon: 0.08 Accuracy: 2255/10000 (23%) 5177 6671  6641 6642 6696 6743 6496  6226
# epsilon: 0.10 Accuracy: 2141/10000 (21%) 4798 6391  6364 6270 6326 6352 6100  5784

# Loss: 1.37855
# epsilon: 0.00 Accuracy: 8804/10000 (88%)
# epsilon: 0.02 Accuracy: 8213/10000 (82%)
# epsilon: 0.04 Accuracy: 7595/10000 (76%)
# epsilon: 0.06 Accuracy: 7051/10000 (71%)
# epsilon: 0.08 Accuracy: 6510/10000 (65%)
# epsilon: 0.10 Accuracy: 6051/10000 (61%)
# Loss: 0.87272                  0.73
# epsilon: 0.00 Accuracy: 8844/10000 (88%)   8756
# epsilon: 0.02 Accuracy: 8372/10000 (84%)   8376
# epsilon: 0.04 Accuracy: 7863/10000 (79%)   7966
# epsilon: 0.06 Accuracy: 7323/10000 (73%)   7530
# epsilon: 0.08 Accuracy: 6736/10000 (67%)   7049
# epsilon: 0.10 Accuracy: 6236/10000 (62%)   6482

# resnet_11                    0.86
# epsilon: 0.00 Accuracy: 8222/10000 (82%) 8630
# epsilon: 0.02 Accuracy: 7805/10000 (78%) 8371
# epsilon: 0.04 Accuracy: 7417/10000 (74%) 8083
# epsilon: 0.06 Accuracy: 7065/10000 (71%) 7780
# epsilon: 0.08 Accuracy: 6707/10000 (67%) 7428
# epsilon: 0.10 Accuracy: 6375/10000 (64%) 7024
