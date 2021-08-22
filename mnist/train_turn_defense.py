import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from advertorch.attacks import FGSM, CarliniWagnerL2Attack, LinfBasicIterativeAttack
import matplotlib.pyplot as plt
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, DefenseNet, reduce_precision_np
from model.model_srresnet import SRResNet

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/checkpoint/resnet50_mnist.pth'))
model1.eval()

model2 = SRResNet().to(device)
model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/checkpoint/defense_turn.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)
epoch = 200


def main():
    model2.train()
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3)
    for i in range(epoch):
        cost = 0
        for img in train_loader:

            data, label = img
            data, label = data.to(device), label.to(device)
            adv_untargeted = adversary_fgsm.perturb(data, label)
            
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 2)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)
           
            adv = []
            for i in range(100):
                angle = 15 * math.pi / 180
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
        torch.save(model2, "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/checkpoint/defense_turn_1.pth")


def tes(eps):
    model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/checkpoint/defense_turn_1.pth')
    model2.eval()

    # adversary_cw = CarliniWagnerL2Attack(model1, num_classes=10, max_iterations=eps,learning_rate=0.1)
    # adversary_if = LinfBasicIterativeAttack(model1,eps = eps,nb_iter=10,eps_iter=0.05)
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    correct = 0
    # a = 0
    for img in test_loader:

        data, label = img
        data, label = data.to(device), label.to(device)
        # 像素缩减
        adv_untargeted = adversary_fgsm.perturb(data, label)
        adv_untargeted = adv_untargeted.cpu().detach().numpy()
        data1 = reduce_precision_np(adv_untargeted, 2)
        data1 = torch.tensor(data1)
        data1 = data1.to(device)
        # 旋转
        adv = []
        for i in range(100):
            angle = 15 * math.pi / 180
            theta = torch.tensor([
                [math.cos(angle), math.sin(-angle), 0],
                [math.sin(angle), math.cos(angle), 0]
            ], dtype=torch.float)

            grid = F.affine_grid(theta.unsqueeze(0), data1[i].unsqueeze(0).size()).to(device)
            output = F.grid_sample(data1[i].unsqueeze(0), grid)
            new_img_torch = output[0]
            adv.append(new_img_torch.tolist())
        adv = torch.tensor(adv).to(device)
        # output = model1(data)
        auto_output = model2(adv)
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
    #     # 初始样本
    #     # plt.subplot(3, size, ii + 1)
    #     # _imshow(data[ii])
    #     # plt.title("Clean \n pred: {}".format(pred_cln[ii]))
    #     # 对抗样本
    #     plt.subplot(3, size, ii + 1)
    #     _imshow(adv_untargeted[ii])
    #     plt.title(" Adv \n pred: {}".format(pred_untargeted_adv[ii]))
    #     # # # bit位压缩
    #     plt.subplot(3, size, ii + 1 + size )
    #     _imshow(data1[ii])
    #     plt.title("Reconstruction \n pred: {}".format(pred_data1_bit[ii]))
    #     # 重建图
    #     plt.subplot(3, size, ii + 1 + size * 2)
    #     _imshow(auto_output[ii])
    #     plt.title("Reconstruction \n pred: {}".format(pred_auto_rec[ii]))

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # main()
    epsilons = [.00, .05, .10, .15, .2, .25, .3]
    for eps in epsilons:
        tes(eps)

#       channel=128  FGSM/BIM/ CW      epoch12_2.075
# epsilon: 0.00 Accuracy: 9940/9940/9940    9924
# epsilon: 0.05 Accuracy: 9183/7578/7114    9899
# epsilon: 0.10 Accuracy: 5954/1401/1190    9865
# epsilon: 0.15 Accuracy: 4772/0787/0349    9827
# epsilon: 0.20 Accuracy: 3680/0635/0271    9754
# epsilon: 0.25 Accuracy: 2806/0565/0174    9676
# epsilon: 0.30 Accuracy: 2273/0525/    9598

#       channel=256             loss_2.079  / 0.4 / 0.3 / 0.17
#                          FGSM/BIMM   FGSM/      BIM/CW
# epsilon: 0.00 Accuracy: 9940/10000 (99%)   9930/9930   9932/  9933  9931/9932
# epsilon: 0.05 Accuracy: 9183/10000 (92%)   9905/9908   9917/  9918  9912/9880
# epsilon: 0.10 Accuracy: 5954/10000 (60%)   9876/9890   9899/  9900  9900/9877
# epsilon: 0.15 Accuracy: 4772/10000 (48%)   9851/9866   9885/  9880  9888/9879
# epsilon: 0.20 Accuracy: 3680/10000 (37%)   9794/9835   9835/  9841  9866/9887
# epsilon: 0.25 Accuracy: 2806/10000 (28%)   9729/9786   9779/  9771  9841/
# epsilon: 0.30 Accuracy: 2273/10000 (23%)   9665/9744   9736/  9728  9802/

# loss 2.61                   2.36  1.11
# epsilon: 0.00 Accuracy: 9934/10000 (99%)  9932  9928
# epsilon: 0.05 Accuracy: 9903/10000 (99%)  9907  9905
# epsilon: 0.10 Accuracy: 9877/10000 (99%)  9888  9883
# epsilon: 0.15 Accuracy: 9862/10000 (99%)  9870  9861
# epsilon: 0.20 Accuracy: 9811/10000 (98%)  9818  9830
# epsilon: 0.25 Accuracy: 9748/10000 (97%)  9763  9776
# epsilon: 0.30 Accuracy: 9684/10000 (97%)  9710  9716