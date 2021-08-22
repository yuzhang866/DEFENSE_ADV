import torch
import torch.nn as nn
import math



class ResidualBlockEdsr(nn.Module):
   
    def __init__(self, n_channel):
        super(ResidualBlockEdsr, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output


class EdsrNet(nn.Module):
    def __init__(self, n_channel):
        super(EdsrNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(ResidualBlockEdsr, 32)

        self.conv_mid = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_output = nn.Conv2d(in_channels=n_channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out, residual)
        out = self.conv_output(out)
        return out
