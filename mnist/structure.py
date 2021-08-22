import torch
import torch.optim as optim
import torchvision
from advertorch.attacks import FGSM,LinfBasicIterativeAttack,CarliniWagnerL2Attack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow
import numpy as np
import torch.nn.functional as F
import sys
from model.model import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/log")


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        #output *= 0.1
        output = torch.add(output, identity_data)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #self.sub_mean = MeanShift(rgb_mean, -1)
        #self.relu = nn.LeakyReLU()

        self.conv1_input = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_input = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_input = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 8)

        self.conv4_output = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_output = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_output = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_output = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        #self.add_mean = MeanShift(rgb_mean, 1)

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
        out = self.conv3_output(out)
        # out = self.relu(out)
        out = self.conv2_output(out)
        # out = self.relu(out)
        out = self.conv1_output(out)
        # out = torch.add(out,residual)

        # out = self.conv_output(out)

        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("==> Preparing data ...")
transform_data = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='../dataset', train=True, download=True, transform=transform_data)
test_data = torchvision.datasets.MNIST(root='../dataset', train=False, download=True,transform=transform_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(torch.load('../SavedNetworkModel/MNIST/ResNetModel/resnet50_mnist.pth', map_location='cpu'))
model1.eval()

model2 = Net().to(device)
model2.train()

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)
epoch = 10


def main():
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3)
    for i in range(epoch):
        print("epoch = ", i)
        cost = 0
        c=0
        name='K'
        for img in train_loader:

            data, label = img
            data, label = data.to(device), label.to(device)

            adv_untargeted = adversary_fgsm.perturb(data, label)
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 2)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)

            auto_output = model2(data1)
            loss = mse_loss(auto_output, data) 

            b = c*100
            writer.add_scalar('loss', loss, b)
            c += 1
            cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.close()
        my_loss = cost / len(train_loader)

        print('epoch: {}, Loss: {:.5f}'.format(i + 1, my_loss))
    writer.close()


if __name__ == '__main__':
    main()


#             Rule/      / ELU        /Lea
# epoch: 1, Loss: 0.00862 0.01425  0.00951 0.00847  0.00846 0.01044
# epoch: 2, Loss: 0.00545 0.00541  0.00726 0.00580  0.00547 0.00546
# epoch: 3, Loss: 0.00516 0.00517  0.00618 0.00545  0.00518 0.00519
# epoch: 4, Loss: 0.00501 0.00503  0.00583 0.00524  0.00503 0.00506
# epoch: 5, Loss: 0.00491 0.00492  0.00563 0.00511  0.00493 0.00496
# epoch: 6, Loss: 0.00484 0.00486  0.00547 0.00500  0.00484 0.00490
# epoch: 7, Loss: 0.00477 0.00482  0.00534 0.00492  0.00479 0.00482
# epoch: 8, Loss: 0.00472 0.00475  0.00524 0.00483  0.00473 0.00474
#               0.00467  0.00514 0.00477  0.00469 0.00468
#               0.00462  0.00509 0.00469  0.00464 0.00467
