# based on source: https://github.com/milesial/Pytorch-UNet
# we add the toric convolution layer

import torch
import torch.nn as nn
import torch.nn.functional as F


class toric_pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        """
        :param x: shape [N, C, H, W]
        :param pad: int >= 0
        """
        x = torch.cat([x, x[:, :, 0:self.pad, :]], dim=2)  # top
        x = torch.cat([x, x[:, :, :, 0:self.pad]], dim=3)  # left
        x = torch.cat([x[:, :, -2 * self.pad:-self.pad, :], x], dim=2)  # left
        x = torch.cat([x[:, :, :, -2 * self.pad:-self.pad], x], dim=3)  # right
        return x


class toroidal_conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding):
        super().__init__()
        assert padding == k // 2, "k: %d, padding: %d" % (k, padding)
        self.pad = padding
        self.toric_pad = toric_pad(padding)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=0)

    def forward(self, x):
        """
        :param x: shape [N, C, H, W]
        :param pad: int >= 0
        :return:
        """
        x = self.toric_pad(x)
        x = self.conv(x)
        return x


class one_conv(nn.Module):
    """ conv => BN => ReLU """
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            toroidal_conv2d(in_ch, out_ch, k=k, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    """ (conv => BN => ReLU) * 2 """
    def __init__(self, in_ch, out_ch, k, padding):
        super().__init__()
        self.conv = nn.Sequential(
            toroidal_conv2d(in_ch, out_ch, k=k, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            toroidal_conv2d(out_ch, out_ch, k=k, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch, k, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.AvgPool2d(2),
            double_conv(in_ch, out_ch, k, padding)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down_one(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.AvgPool2d(2),
            one_conv(in_ch, out_ch, k, padding)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1, bilinear=True):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch, k=k, padding=padding)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='nearest')#, align_corners=True)
        # TODO: in case of different stride/pads etc where the input images have different sizes
        # TODO: we need to pad it torically some how. it might just work as is, needs to check...
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up_one(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1, bilinear=True):
        super().__init__()
        self.conv = one_conv(in_ch, out_ch, k=k, padding=padding)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='nearest')#, align_corners=True)
        # TODO: in case of different stride/pads etc where the input images have different sizes
        # TODO: we need to pad it torically some how. it might just work as is, needs to check...
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


####################################################################################################
####################################################################################################
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNetDeep(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = inconv(n_channels, 128, k=5, padding=2)
        self.down1 = down(128, 128)
        self.down2 = down_one(128, 128)
        self.down3 = down_one(128, 256)
        self.down4 = down_one(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up_one(512, 128)
        self.up3 = up_one(256, 128)
        self.up4 = up_one(256, 128)
        self.up5 = up_one(256, 128)
        self.outc = outconv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x
