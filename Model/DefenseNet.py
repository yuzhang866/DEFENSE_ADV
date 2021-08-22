import torch.nn as nn
import numpy as np


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):

        super(ConvolutionalBlock, self).__init__()
        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}
        
        layers = list()
        
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))
        
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

       
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)
        return output


class ResidualBlock(nn.Module):
    
    def __init__(self, kernel_size=3, n_channels=64):

        super(ResidualBlock, self).__init__()
       
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')
       
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class DefenseNet(nn.Module):
    """
    SRResNet Model
    """
    def __init__(self, large_kernel_size=7, small_kernel_size=3, n_channels=64, n_blocks=24, scaling_factor=None):

        super(DefenseNet, self).__init__()
        
        # scaling_factor = int(2)
        # assert scaling_factor in {2, 4, 8}, 

        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation='prelu')
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

       
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

       
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
       
        output = self.conv_block1(lr_imgs)
        residual = output                                           # (batch_size, 64, 14, 14)
        output = self.residual_blocks(output)                       # (batch_size, 64, 14, 14)
        output = self.conv_block2(output)                           # (batch_size, 64, 14, 14)
        output = output + residual                                  # (batch_size, 64, 14, 14)
        sr_imgs = self.conv_block3(output)                          # (batch_size, 1, 28, 28)
        return sr_imgs



def reduce_precision_np(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float
