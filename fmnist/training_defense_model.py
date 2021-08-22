import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from advertorch.attacks import FGSM
from torchvision import transforms


############################# resnet50 #####################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


######################################### defense model ###################################################

class _Residual_Block(nn.Module):  # CONV+RELU+CONV
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        # output *= 0.1
        output = torch.add(output, identity_data)
        return output


#########################################################################################################
class COM_Net(nn.Module):
    def __init__(self):
        super(COM_Net, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        # self.relu = nn.ReLU()

        self.conv1_input = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_input = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_input = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 8)

        self.conv4_output = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_output = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_output = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_output = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

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
        out = self.conv3_output(out)
        # out = self.relu(out)
        out = self.conv2_output(out)
        # out = self.relu(out)
        out = self.conv1_output(out)
        out = torch.sigmoid(out)

        # out = torch.add(out,residual)

        # out = self.conv_output(out)

        return out


#########################################################################################################
class REC_Net(nn.Module):
    def __init__(self):
        super(REC_Net, self).__init__()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        # self.relu = nn.ReLU()

        self.conv1_input = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_input = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_input = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 8)

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
        out = self.conv3_output(out)
        # out = self.relu(out)
        out = self.conv2_output(out)
        # out = self.relu(out)
        out = self.conv1_output(out)
        # out = torch.add(out,residual)

        # out = self.conv_output(out)

        return out


##########################################################################################


###############################################
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


######################################### 参数设置 ###################################################
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform_data = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform_data)
test_data = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True, num_workers=2)

model1 = ResNet50().to(device)
# model1 = torch.load("/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/FMNIST/checkpoint/Fashion_resnet50.pth")
model1.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/F-MNIST/checkpoint_resnet50/resnet50_11.pth'))
model1.eval()

model2 = COM_Net().to(device)
model3 = REC_Net().to(device)

awl = AutomaticWeightedLoss(2).to(device)
loss1 = nn.MSELoss()
loss2 = nn.L1Loss()
# PATH = ('/content/drive/My Drive/Colab Notebooks/DEFENSE_ADV/10月FMNIST/checkpoint/defense.pth')

# optimizer = optim.Adam([{'params': model2.parameters()},
#              {'params': model3.parameters()},
#              {'params': awl.parameters(), 'weight_decay': 0}],lr=0.0001)
epoch = 300


#########################################################################################################
def main():
    adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.015)
    for i in range(epoch):
        cost = 0
        for img in train_loader:
            # forward
            # pred1, pred2 = Model1(data)
            # calculate losses
            data, label = img
            data, label = data.to(device), label.to(device)
            adv_untargeted = adversary_fgsm.perturb(data, label)
            com_output = model2(adv_untargeted)
            rec_output = model3(com_output)

            loss_1 = loss1(data, com_output)
            loss_2 = loss2(data, rec_output)
            # weigh losses
            loss_sum = awl(loss_1, loss_2)
            cost += loss_sum.item()
            # backward
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

        my_loss = cost / len(train_loader)
        print('epoch: {}, Loss: {:.5f}'.format(i + 1, my_loss))
        torch.save({
            'com_model': model2.state_dict(),
            'rec_model': model3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)

        # correct1=0
        epsilons = [.00, .02, .04, .06, .08, .1]
        for eps in epsilons:
            correct2 = 0
            for img in test_loader:
                data, label = img
                data, label = data.to(device), label.to(device)

                adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
                adv_untargeted = adversary_fgsm.perturb(data, label)
                com_output = model2(adv_untargeted)
                rec_output = model3(com_output)

                # com = model1(com_output)
                rec = model1(rec_output)

                # pred1 = com.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # correct1 += pred1.eq(label.data.view_as(pred1)).cpu().sum()  

                pred2 = rec.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(label.data.view_as(pred2)).cpu().sum()  

            # print('epoch: {}, Accuracy: {}/{} ({:.0f}%)'.format(i + 1,correct1, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            print('epoch: {}, Accuracy: {}/{} ({:.0f}%)'.format(i + 1, correct2, len(test_loader.dataset),
                                                                100. * correct2 / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
