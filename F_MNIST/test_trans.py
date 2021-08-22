import torch
import torch.optim as optim
import torchvision
from advertorch.attacks import FGSM,LinfBasicIterativeAttack,CarliniWagnerL2Attack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, DefenseNet, reduce_precision_np
from model.model import resnet101
from model_srresnet_128 import SRResNet

################################## LeNet ########################################
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

############################### GOOGLENET ####################################
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

net = nn.Sequential(b1, b2, b3, b4, b5,
                    FlattenLayer(), nn.Linear(1024, 10))
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
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_data)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True,transform=transform_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True, num_workers=4)


model1 = resnet50(10,1).to(device)
model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet50_11.pth'))
model1.eval()

# model1 = resnet101(10,1).to(device)
# model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet101_8.pth'))
# model1.eval()

# model1 = net.to(device)
# model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_GoogLeNet/net_2.pth'))
# model1.eval()

# model1 = Cnn(1,10).to(device)
# model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_LeNet/Lenet_fmnist_7.pth'))
# model1.eval()

model2 = SRResNet().to(device)
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense35.pth')
# model2.eval()

def tes(eps):
    #model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/4bit_128/defense136.pth')
    #model2.eval()

    adversary_cw = CarliniWagnerL2Attack(model1, num_classes=10, max_iterations=eps,learning_rate=0.1)
    #adversary_if = LinfBasicIterativeAttack(model1,eps = eps,nb_iter=10,eps_iter=0.05)
    #adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
    correct = 0
    a = 0
    for img in test_loader:

        # a += 1
        # print(a)

        data, label = img
        data, label = data.to(device), label.to(device)

        adv_untargeted = adversary_cw.perturb(data, label)
        # adv_untargeted = adv_untargeted.cpu().detach().numpy()
        # data1 = reduce_precision_np(adv_untargeted, 4)
        # data1 = torch.tensor(data1)
        # data1 = data1.to(device)

        output = model1(adv_untargeted)
        # auto_output = model2(data1)
        # output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps,correct, len(test_loader.dataset),
                                100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    #epsilons = [.00, .02, .05, .06, .08, .1]
    #epsilons = [30]
    epsilons = [10,20,30,40,50]
    for eps in epsilons:
        tes(eps)

# LeNet           FGSM/BIMM/CW
# epsilon: 0.00 Accuracy: 8974/8974/0067
# epsilon: 0.02 Accuracy: 5843/5878/0000
# epsilon: 0.04 Accuracy: 3437/2871/0000
# epsilon: 0.06 Accuracy: 2033/0853/0000
# epsilon: 0.08 Accuracy: 1281/0094/0000
# epsilon: 0.10 Accuracy: 0920/0002/0000

# GoogLeNet         FGSM/BIMM/CW
# epsilon: 0.00 Accuracy: 9068/8984
# epsilon: 0.02 Accuracy: 6281/8631/10000 (86%)
# epsilon: 0.05 Accuracy: 3545/8128/10000 (81%)
# epsilon: 0.06 Accuracy: 2955/7984/10000 (80%)
# epsilon: 0.08 Accuracy: 2105/7605/10000 (76%)
# epsilon: 0.10 Accuracy: 1554/7151/10000 (72%)

# ResNet101_8        FGSM/BIMM/CW    FGSM/BIMM/CW
# epsilon: 0.00 Accuracy: 9270/9270         8825
# epsilon: 0.02 Accuracy: 5345/4956         8664
# epsilon: 0.05 Accuracy: 1990/0194         8361
# epsilon: 0.06 Accuracy: 1729/0152         8285
# epsilon: 0.08 Accuracy: 1291/0152         8034
# epsilon: 0.10 Accuracy: 1025/0152         7781
