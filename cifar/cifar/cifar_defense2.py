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

import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/model')
sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model import resnet50, DefenseNet, reduce_precision_np
from model_srresnet_128_cifar import SRResNet


################################### ResNet50 ################################
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


###################################################################

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=30, shuffle=False, num_workers=0)

model1 = ResNet(50, 10).to(device)
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/checkpoint/cifar_resnet.pth'))
model1.eval()

model2 = SRResNet().to(device)
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/checkpoint/cifar_defense2.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)

epoch = 100


def main():
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.04)

    for i in range(epoch):
        correct = 0
        cost = 0
        for img in train_loader:
            data, label = img
            data, label = data.to(device), label.to(device)

            adv_untargeted = adversary_fgsm.perturb(data, label)
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 10)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)

            auto_output = model2(data1).to(device)
            loss = mse_loss(auto_output, data)  # 平均损失

            cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, Auto-Loss: {:.5f}'.format(i + 1, cost))
        torch.save(model2, "/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/checkpoint/cifar_defense4.pth")

        epsilons = [.00, .002, .004, .006, .008, .01]
        for eps in epsilons:
            correct = 0
            for img in test_loader:
                adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
                model2.eval()
                data, label = img
                data, label = data.to(device), label.to(device)
                adv_untargeted = adversary_fgsm.perturb(data, label)
                adv_untargeted = adv_untargeted.cpu().detach().numpy()
                data1 = reduce_precision_np(adv_untargeted, 10)
                data1 = torch.tensor(data1)
                data1 = data1.to(device)
                auto_output = model2(data1)
                output = model1(auto_output)

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
            print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))


def tes(eps):
    model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/checkpoint/cifar_defense4.pth')
    model2.eval()

    # adversary_if = LinfBasicIterativeAttack(model1,eps = eps,nb_iter=10,eps_iter=0.05)
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
        data1 = reduce_precision_np(adv_untargeted, 10)
        data1 = torch.tensor(data1)
        data1 = data1.to(device)

        # output = model1(data1)
        auto_output = model2(data1)
        output = model1(auto_output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                         100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # main()
    epsilons = [.006]
    for eps in epsilons:
        tes(eps)

# e=0.02            FGSM       _8bit/10bit/12bit  defense0.286
# epsilon: 0.000 Accuracy: 9298/10000 (93%)  8205/8586/8800     8958
# epsilon: 0.002 Accuracy: 6712/10000 (67%)  7558/7639/7907     8460
# epsilon: 0.004 Accuracy: 4790/10000 (48%)  7063/7107/7086     8111
# epsilon: 0.006 Accuracy: 3667/10000 (37%)  6296/6049/5974     7569
# epsilon: 0.008 Accuracy: 3040/10000 (30%)  5728/5546/5180     7168
# epsilon: 0.010 Accuracy: 2584/10000 (26%)  5032/4600/4299     6530
# 10bit e=0.02 loss0.31              e=0.03 loss0.28
# epsilon: 0.000 Accuracy: 8935/10000 (89%)     8831
# epsilon: 0.002 Accuracy: 8305/10000 (83%)     8315
# epsilon: 0.004 Accuracy: 7957/10000 (80%)
# epsilon: 0.006 Accuracy: 7199/10000 (72%)
# epsilon: 0.008 Accuracy: 6832/10000 (68%)
# epsilon: 0.010 Accuracy: 6074/10000 (61%)     6483
# epsilon: 0.020 Accuracy: 4054/10000 (41%)
