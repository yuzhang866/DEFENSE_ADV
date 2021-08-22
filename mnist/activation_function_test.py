import sys
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from advertorch.attacks import FGSM
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/model')
from model.model import resnet50, reduce_precision_np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/MNIST/log")


###############################################################################

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

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1_input(x)
        out = self.relu(out)
        out = self.conv2_input(out)
        out = self.relu(out)
        out = self.conv3_input(out)
        out = self.relu(out)
        out = self.conv4_input(out)
        out = self.relu(out)
        # residual = out
        out = self.conv4_output(self.residual(out))
        out = self.relu(out)
        out = self.conv3_output(out)
        out = self.relu(out)
        out = self.conv2_output(out)
        out = self.relu(out)
        out = self.conv1_output(out)
        # out = torch.add(out,residual)

        # out = self.conv_output(out)

        return out


###############################################################################
device = ''
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. GPU will be used for training.")
else:
    device = 'cpu'

print("==> Prepairing data ...")
transform_data = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=transform_data)
test_data = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=transform_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True, num_workers=4)

model1 = resnet50(10, 1).to(device)
model1.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/checkpoint/resnet50_mnist.pth'))
model1.eval()

model2 = Net().to(device)

mse_loss = nn.MSELoss().cuda()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)
epoch = 10


def main():
    adversary_fgsm1 = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3)
    adversary_fgsm2 = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15)
    for i in range(epoch):
        # correct = 0
        # cost=0
        for img in train_loader:
            data, label = img
            data, label = data.to(device), label.to(device)

            adv_untargeted = adversary_fgsm1.perturb(data, label)
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 2)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)

            defense_output = model2(data1)
            loss = mse_loss(defense_output, data)
            # cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output = model1(defense_output)

            # pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # correct += pred.eq(label.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        # print('epoch:{}, Accuracy: {}/{} ({:.0f}%)'.format(i+1,correct, len(train_loader.dataset),
        #                            100. * correct / len(train_loader.dataset)))
        # print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps,correct, len(test_loader.dataset),
        #                             100. * correct / len(test_loader.dataset)))
        correct = 0
        for img in test_loader:
            model2.eval()
            data, label = img
            data, label = data.to(device), label.to(device)

            adv_untargeted = adversary_fgsm2.perturb(data, label)
            adv_untargeted = adv_untargeted.cpu().detach().numpy()
            data1 = reduce_precision_np(adv_untargeted, 2)
            data1 = torch.tensor(data1)
            data1 = data1.to(device)

            defense_output = model2(data1)
            output = model1(defense_output)

            pred = output.data.max(1, keepdim=True)[1]  # get the index of theest max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum() 
        b = 100. * correct / len(test_loader.dataset)
        writer.add_scalar('Accuracy', b, i + 1)
        print('epoch:{}, Accuracy: {}/{} ({:.0f}%)'.format(i + 1, correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))

    writer.close()


if __name__ == '__main__':
    main()

# relu1
#  93%
#  96%
#