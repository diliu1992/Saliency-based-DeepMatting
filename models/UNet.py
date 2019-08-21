import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo


class UNet(nn.Module):
    def __init__(self, n_classes=21, pretrained=False):
        super(UNet, self).__init__()
        self.down1 = unetDown(in_channels=3, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = unetDown(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = unetDown(in_channels=128, out_channels=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = unetDown(in_channels=256, out_channels=512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = unetDown(in_channels=512, out_channels=1024)

        self.up4 = unetUp(in_channels=1024, out_channels=512)
        self.up3 = unetUp(in_channels=512, out_channels=256)
        self.up2 = unetUp(in_channels=256, out_channels=128)
        self.up1 = unetUp(in_channels=128, out_channels=64)

        self.classifier = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        out_size = x.size()[2:]
        down1_x = self.down1(x)
        maxpool1_x = self.maxpool1(down1_x)
        # print('maxpool1_x.data.size():', maxpool1_x.data.size())
        down2_x = self.down2(maxpool1_x)
        maxpool2_x = self.maxpool2(down2_x)
        # print('maxpool2_x.data.size():', maxpool2_x.data.size())
        down3_x = self.down3(maxpool2_x)
        maxpool3_x = self.maxpool3(down3_x)
        # print('maxpool3_x.data.size():', maxpool3_x.data.size())
        down4_x = self.down4(maxpool3_x)
        maxpool4_x = self.maxpool1(down4_x)
        # print('maxpool4_x.data.size():', maxpool4_x.data.size())

        center_x = self.center(maxpool4_x)
        # print('center_x.data.size():', center_x.data.size())

        up4_x = self.up4(center_x, down4_x)
        # print('up4_x.data.size():', up4_x.data.size())
        up3_x = self.up3(up4_x, down3_x)
        # print('up3_x.data.size():', up3_x.data.size())
        up2_x = self.up2(up3_x, down2_x)
        # print('up2_x.data.size():', up2_x.data.size())
        up1_x = self.up1(up2_x, down1_x)
        # print('up1_x.data.size():', up1_x.data.size())

        x = self.classifier(up1_x)
        # 最后将模型上采样到原始分辨率
        x = F.upsample_bilinear(x, out_size)

        return x


class unetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetDown, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetUp, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_cur, x_prev):
        x = self.upConv(x_cur)
        x = torch.cat([F.upsample_bilinear(x_prev, size=x.size()[2:]), x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
