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
import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, DefenseNet, reduce_precision_np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        # output *= 0.1
        output = torch.add(output, identity_data)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.relu = nn.ReLU()

        self.conv1_input = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_input = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_input = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv4_output = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_output = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_output = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_output = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        # self.add_mean = MeanShift(rgb_mean, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1_input(x)
        # out = self.relu(out)
        out = self.conv2_input(out)
        # out = self.relu(out)
        out = self.conv3_input(out)
        # out = self.relu(out)
        out = self.conv4_input(out)
        # out = self.relu(out)
        # residual = out
        out = self.conv4_output(self.residual(out))
        # out = self.relu(out)
        # out = torch.add(out,residual)
        out = self.conv3_output(out)
        # out = self.relu(out)
        out = self.conv2_output(out)
        # out = self.relu(out)
        out = self.conv1_output(out)
        # out = torch.add(out,residual)

        # out = self.conv_output(out)

        return out


##############################################################
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

model2 = Net().to(device)
# model2 = torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn.pth')

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.00003)
epoch = 1


def main():
    model2.train()
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1)
    for epoch in range(10):
        cost = 0
        b = 0
        # a=0
        correct = 0
        for img in train_loader:
            # a +=1
            # print(a)
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

            output = model1(auto_output)

            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()  

            # writer.add_scalar('loss', loss, b)

            b += 100
        writer.add_scalar('acc', 100. * correct / len(test_loader.dataset), epoch + 1)
        print('epoch: {}, Loss: {:.5f}'.format(epoch + 1, cost))
        # torch.save(model2,"/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint/defense_turn_1.pth")
    writer.close()


if __name__ == '__main__':
    main()
