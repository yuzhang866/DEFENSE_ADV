import torch
import torch.optim as optim
import torchvision
from advertorch.attacks import FGSM, LinfBasicIterativeAttack, CarliniWagnerL2Attack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow

import matplotlib.pyplot as plt

from model.model import resnet50, reduce_precision_np
from model.model_edsr import Net
from model.model_srresnet_128_not import SRResNet

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox as fb

################################## LeNet ########################################
# class Cnn(nn.Module):
#     def __init__(self, in_dim, n_class):
#         super(Cnn, self).__init__()   
#         self.conv = nn.Sequential(    
#             nn.Conv2d(in_dim, 6, 5, stride=1, padding=2),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, 5, stride=1, padding=0),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2))

#         self.fc = nn.Sequential(
#             nn.Linear(400, 120),
#             nn.Linear(120, 84),
#             nn.Linear(84, n_class))

#     def forward(self, x):
#         out = self.conv(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# ############################### GOOGLENET ####################################
# class Inception(nn.Module):
#    
#     def __init__(self, in_c, c1, c2, c3, c4):
#         super(Inception, self).__init__()
#       
#         self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
#         
#         self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
#         self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
#        
#         self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
#         self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
#       
#         self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

#     def forward(self, x):
#         p1 = F.relu(self.p1_1(x))
#         p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
#         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
#         p4 = F.relu(self.p4_2(self.p4_1(x)))
#         return torch.cat((p1, p2, p3, p4), dim=1)  

# b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
#                    nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
#                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
#                    Inception(256, 128, (128, 192), (32, 96), 64),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
#                    Inception(512, 160, (112, 224), (24, 64), 64),
#                    Inception(512, 128, (128, 256), (24, 64), 64),
#                    Inception(512, 112, (144, 288), (32, 64), 64),
#                    Inception(528, 256, (160, 320), (32, 128), 128),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# class GlobalAvgPool2d(nn.Module):
#    
#     def __init__(self):
#         super(GlobalAvgPool2d, self).__init__()

#     def forward(self, x):
#         return F.avg_pool2d(x, kernel_size=x.size()[2:])


# class FlattenLayer(torch.nn.Module):
#     def __init__(self):
#         super(FlattenLayer, self).__init__()

#     def forward(self, x):  # x shape: (batch, *, *, ...)
#         return x.view(x.shape[0], -1)


# b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
#                    Inception(832, 384, (192, 384), (48, 128), 128),
#                    GlobalAvgPool2d())

# net = nn.Sequential(b1, b2, b3, b4, b5,
#                     FlattenLayer(), nn.Linear(1024, 10))

#################################################################################################

device = ''
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. GPU will be used for training.")
else:
    device = 'cpu'

BEST_ACCURACY = 0

# Preparing Data
print("==> Prepairing data ...")
transform_data = transforms.Compose([transforms.ToTensor()])
# train_data = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=transform_data)
test_data = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=transform_data)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/checkpoint/resnet50_mnist.pth'))
model1.eval()

# model1 = net.to(device)
# model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/ck_googlenet/googlenet_mnist_6.pth'))
# model1.eval()

# model1 = Cnn(1,10).to(device)
# model1.load_state_dict(torch.load('/DEFENSE_ADV/MNIST/ck_googlenet/Lenet_mnist_10.pth'))
# model1.eval()

model2 = SRResNet().to(device)


# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/checkpoint/defense0.3_256.pth')
# model2.eval()

def tes(eps):
    model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense50.pth')
    model2.eval()
    # adversary_cw = CarliniWagnerL2Attack(model1, num_classes=10, max_iterations=eps,learning_rate=0.1)
    # adversary_if = LinfBasicIterativeAttack(model1,eps = eps,nb_iter=10,eps_iter=0.05)
    # adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    correct = 0
    a = 0
    for img in test_loader:
        a += 1
        print(a)

        data, label = img
        data, label = data.to(device), label.to(device)

        ################foolbox##########################
        bounds = (0, 1)
        fmodel = fb.PyTorchModel(model1, bounds)
        attack = fb.attacks.LinfDeepFoolAttack()
        raw, clipped, adv = attack(fmodel, data, label, epsilons=eps)

        adv_untargeted = clipped.cpu().detach().numpy()
        data1 = reduce_precision_np(adv_untargeted, 2)
        data1 = torch.tensor(data1)
        data1 = data1.to(device)

        #################################################

        # adv_untargeted = adversary_cw.perturb(data, label)
        # adv_untargeted = adv_untargeted.cpu().detach().numpy()
        # data1 = reduce_precision_np(adv_untargeted, 2)
        # data1 = torch.tensor(data1)
        # data1 = data1.to(device)

        # output = model1(clipped)
        auto_output = model2(data1)
        output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))

    data = data.to(device)
    adv_untargeted = torch.tensor(adv_untargeted).to(device)

    pred_cln = predict_from_logits(model1(data))
    pred_untargeted_adv = predict_from_logits(model1(adv_untargeted))
    pred_data1_bit = predict_from_logits(model1(data1))
    pred_auto_rec = predict_from_logits(model1(auto_output))

    size = 10
    plt.figure(figsize=(20, 8))
    for ii in range(10):
       
        plt.subplot(4, size, ii + 1)
        _imshow(data[ii])
        plt.title("Clean \n pred: {}".format(pred_cln[ii]))
        
        plt.subplot(4, size, ii + 1 + size)
        _imshow(adv_untargeted[ii])
        plt.title(" Adv \n pred: {}".format(pred_untargeted_adv[ii]))
       
        plt.subplot(4, size, ii + 1 + size * 2)
        _imshow(data1[ii])
        plt.title("Reconstruction \n pred: {}".format(pred_data1_bit[ii]))
       
        plt.subplot(4, size, ii + 1 + size * 3)
        _imshow(auto_output[ii])
        plt.title("Reconstruction \n pred: {}".format(pred_auto_rec[ii]))

    plt.tight_layout()
    plt.savefig('./fig/bim.png', dpi=600, pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    # epsilons = [.00, .05, .10, .15, .2, .25, .3]
    epsilons = [.2]
    for eps in epsilons:
        tes(eps)
#                       color_reduce
# GoogLeNet         FGSM/BIMM/CW    k=2 /3 / 4        with defense
# epsilon: 0.00 Accuracy: 9851/9851/9851   9899 9924 9935       9864/9864/9864
# epsilon: 0.05 Accuracy: 9007/8753/7388   9818 9863 9848        9805/9823/9753
# epsilon: 0.10 Accuracy: 6769/4517/3007   9698 9708 9681       9744/9753/9736
# epsilon: 0.15 Accuracy: 4761/2578/1548   9533 9452 9378        9685/9731/9756
# epsilon: 0.20 Accuracy: 3705/2241/0930   9257 9008 2191        9617/9690/9743
# epsilon: 0.25 Accuracy: 3229/2097/0639   8895 7028 2144        9516/9638/9777
# epsilon: 0.30 Accuracy: 3039/2009/0000   8441 1579 2105       9413/9587/
# LeNet             BIM
# epsilon: 0.00 Accuracy: 9888/9886/
# epsilon: 0.05 Accuracy: 8700/9836/
# epsilon: 0.10 Accuracy: 3052/9798/
# epsilon: 0.15 Accuracy: 0172/9753/
# epsilon: 0.20 Accuracy: 0005/9630/
# epsilon: 0.25 Accuracy: 0000/9535/
# epsilon: 0.30 Accuracy: 0000/9401/


# DEEPFOOL
# epsilon: 0.15 Accuracy: 1818/10000 (18%)