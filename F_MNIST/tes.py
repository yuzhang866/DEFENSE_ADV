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
from advertorch.attacks import FGSM, CarliniWagnerL2Attack, LinfBasicIterativeAttack, JacobianSaliencyMapAttack
import matplotlib.pyplot as plt
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow
import foolbox as fb
import torchattacks
import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')

from model.model import resnet50, DefenseNet, reduce_precision_np
from model_srresnet_128 import SRResNet

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox as fb

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet50_11.pth'))
model1.eval()

model2 = SRResNet().to(device)
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.00003)
epoch = 1


def main():
    model2.train()
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1)
    for i in range(epoch):
        cost = 0
        b = 0
        for img in train_loader:
            data, label = img
            data, label = data.to(device), label.to(device)
            adv_untargeted = adversary_fgsm.perturb(data, label)
            
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 4)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)

            
            auto_output = model2(data1)
            loss = mse_loss(auto_output, data)  

            cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # c = b*100
            writer.add_scalar('loss', loss, b)
            b += 100

        print('epoch: {}, Loss: {:.5f}'.format(i + 1, cost))
        torch.save(model2, "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn_1.pth")
    writer.close()


def tes(eps):
    model2 = torch.load(
        '/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense110.pth')
    model2.eval()

    # adversary_cw = CarliniWagnerL2Attack(model1,10,max_iterations=eps)
    # adversary_if = LinfBasicIterativeAttack(model1,eps = eps,nb_iter=20,eps_iter=0.05)
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    # adversary_jsma = JacobianSaliencyMapAttack(model1,10,theta=1.0,gamma=1.0)
    correct = 0
    a = 0
    for img in test_loader:
        a += 1
        print(a)
        data, label = img
        data, label = data.to(device), label.to(device)
        

        ##########################
        # bounds = (0, 1)
        # fmodel = fb.PyTorchModel(model1,bounds)
        # attack = fb.attacks.LinfDeepFoolAttack()
        # raw, adv_untargeted, adv = attack(fmodel, data, label, epsilons=eps)

        # adv_untargeted = adv_untargeted.cpu().detach().numpy()
        data1 = reduce_precision_np(adv_untargeted, 4)
        # data1 = torch.tensor(data1)
        # data1 = data1.to(device)

        ##########################
        adv_untargeted = adversary_fgsm.perturb(data, label)
        adv_untargeted = adv_untargeted.cpu().detach().numpy()
        data1 = reduce_precision_np(adv_untargeted, 4)
        data1 = torch.tensor(data1).to(device)

        # output = model1(adv_untargeted)
        auto_output = model2(data1)
        output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))

    # data = data.to(device)
    # adv_untargeted = torch.tensor(adv_untargeted).to(device)

    # pred_cln = predict_from_logits(model1(data))
    # pred_untargeted_adv = predict_from_logits(model1(adv_untargeted))
    # pred_data1_bit = predict_from_logits(model1(data1))
    # pred_auto_rec = predict_from_logits(model1(auto_output))

    # size = 10
    # plt.figure(figsize=(20, 8))
    # for ii in range(10):
    #    
    #     plt.subplot(4, size, ii + 1)
    #     _imshow(data[ii])
    #     plt.title("Clean \n pred: {}".format(pred_cln[ii]))
    #     
    #     plt.subplot(4, size, ii + 1 + size)
    #     _imshow(adv_untargeted[ii])
    #     plt.title(" Adv \n pred: {}".format(pred_untargeted_adv[ii]))
    #
    #     plt.subplot(4, size, ii + 1 + size * 2)
    #     _imshow(data1[ii])
    #     plt.title("Compression \n pred: {}".format(pred_data1_bit[ii]))
    #     
    #     plt.subplot(4, size, ii + 1 + size * 3)
    #     _imshow(auto_output[ii])
    #     plt.title("Reconstruction \n pred: {}".format(pred_auto_rec[ii]))

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
    # epsilons = [40]
    # epsilons = [3,5,8,10]
    # epsilons = [.07,.02,.04,.06,.08,.1]
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

# turn11 10  9   8  7777  6   5
# epsilon: 0.00 Accuracy: 9308/10000 (93%) 8595 7692  7838 8002 8128 8243 8335  8404
# epsilon: 0.02 Accuracy: 4384/10000 (44%) 7461 7352  7502 7638 7708 7787 7796  7740
# epsilon: 0.04 Accuracy: 2806/10000 (28%) 6484 7140  7191 7274 7360 7436 7337  7176
# epsilon: 0.06 Accuracy: 2474/10000 (25%) 5703 6917  6921 6965 7062 7081 6914  6655
# epsilon: 0.08 Accuracy: 2255/10000 (23%) 5177 6671  6641 6642 6696 6743 6496  6226
# epsilon: 0.10 Accuracy: 2141/10000 (21%) 4798 6391  6364 6270 6326 6352 6100  5784
