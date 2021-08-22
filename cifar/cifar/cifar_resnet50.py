import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from advertorch.attacks import FGSM


# import foolbox as fb
# import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
# from model import resnet50


#############################################################################
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


###########################################################
device = ''
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. GPU will be used for training.")
else:
    device = 'cpu'

BEST_ACCURACY = 0
print("==> Prepairing data ...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

model1 = ResNet(50, 10).to(device)
# model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/checkpoint/cifar_resnet.pth'))

optimizer = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = nn.CrossEntropyLoss()

length_train = len(train_data)
length_validation = len(test_data)
num_classes = 10


def train(epochs):
    global BEST_ACCURACY
    dict = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}
    a = 0
    for epoch in range(epochs):
        print("\nEpoch:", epoch + 1, "/", epochs)
        cost = 0
        correct = 0
        total = 0
        woha = 0

        for i, (x, y) in enumerate(train_loader):
            woha += 1
            model1.train()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            yhat = model1(x)
            yhat = yhat.reshape(-1, 10)

            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            cost += loss.item()

            _, yhat2 = torch.max(yhat.data, 1)
            correct += (yhat2 == y).sum().item()
            total += y.size(0)
        scheduler.step()

        my_loss = cost / len(train_loader)
        my_accuracy = 100 * correct / length_train

        dict['Train Loss'].append(my_loss)
        dict['Train Acc'].append(my_accuracy)

        print('Tain Loss:', my_loss)
        print('Train Accuracy:', my_accuracy, '%')

        cost = 0
        correct = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                model1.eval()
                yhat = model1(x)
                yhat = yhat.reshape(-1, 10)
                loss = criterion(yhat, y)
                cost += loss.item()

                _, yhat2 = torch.max(yhat.data, 1)
                correct += (yhat2 == y).sum().item()

        my_loss = cost / len(test_loader)
        my_accuracy = 100 * correct / length_validation

        dict['Validation Loss'].append(my_loss)
        dict['Validation Acc'].append(my_accuracy)

        print('Validation Loss:', my_loss)
        print('Validation Accuracy:', my_accuracy, '%')

        # torch.save(model1.state_dict(), '/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/new_resnet_{}.pth'.format(a+1))
        # a += 1
        # Save the model if you get best accuracy on validation data
        if my_accuracy > BEST_ACCURACY:
            BEST_ACCURACY = my_accuracy
            print('Saving the model ...')
            model1.eval()
            torch.save(model1.state_dict(),
                       '/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/CIFAR/checkpoint/cifar_resnet_915.pth')
        # epsilons = [.002,.008,.01]
        # for eps in epsilons:
        #   correct = 0
        #   for x, y in test_loader:
        #       x, y = x.to(device), y.to(device)
        #       model1.eval()

        #       fmodel = fb.PyTorchModel(model1, bounds=(0, 1))
        #       attack = fb.attacks.FGSM()
        #       epsilons = eps
        #       _, advs, success = attack(fmodel, x, y, epsilons=epsilons)

        #       # adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
        #       # adv_untargeted = adversary_fgsm.perturb(x, y)
        #       output = model1(advs)
        #       pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #       correct += pred.eq(y.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        #   print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps,correct, len(test_loader.dataset),
        #                     100. * correct / len(test_loader.dataset)))

    print("TRAINING IS FINISHED !!!")
    return dict


# Start training
results = train(200)
