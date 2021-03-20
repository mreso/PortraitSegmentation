import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.shufflenetv2 import channel_shuffle


class BnPRelu(nn.Module):
    def __init__(self, num_input):
        super().__init__()

        self.bn = nn.BatchNorm2d(num_input, eps=0.001)
        self.prelu = nn.PReLU(num_input)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)

        return x


class ConvBnPRelu(nn.Module):
    def __init__(self, num_input, num_output, kernel_size, stride):
        super().__init__()
        padding = int((kernel_size-1)/2)

        self.conv = nn.Conv2d(num_input, num_output,
                              kernel_size, stride, bias=False, padding=padding)
        self.bn = nn.BatchNorm2d(num_output, eps=0.001)
        self.prelu = nn.PReLU(num_output)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x


class SE(nn.Module):
    def __init__(self, num_input, reduction=2):
        super().__init__()
        num_reduced_channels = num_input // reduction
        if reduction > 1:
            self.fc = nn.Sequential(
                nn.Linear(num_input, num_reduced_channels),
                nn.PReLU(num_reduced_channels),
                nn.Linear(num_reduced_channels, num_input)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_input, num_input),
                nn.PReLU(num_input),
            )

    def forward(self, x):
        batch, num_channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(batch, num_channels)
        y = self.fc(y).view(batch, num_channels, 1, 1)
        return x * y.expand_as(x)


class DSConvSE(nn.Module):
    def __init__(self, num_input, num_output, kernel_size, stride, reduction=2):
        super().__init__()
        padding = int((kernel_size-1)/2)
        self.depthwise_conv = nn.Conv2d(
            num_input, num_input, kernel_size, stride=stride, groups=num_input, bias=False, padding=padding)
        self.se = SE(num_input, reduction=reduction)
        self.pointwise_conv = nn.Conv2d(num_input, num_output, 1, bias=False)

        self.bn = nn.BatchNorm2d(num_output, eps=0.001)
        self.prelu = nn.PReLU(num_output)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)

        x = self.bn(x)
        x = self.prelu(x)

        return x


class S2Block(nn.Module):
    def __init__(self, num_input, config):
        super().__init__()
        kernel_size, self.pooling = config

        padding = (kernel_size-1)//2

        if self.pooling > 1:
            self.down_sample = nn.AvgPool2d(self.pooling, self.pooling)
            self.up_sample = nn.UpsamplingBilinear2d(scale_factor=self.pooling)

        self.conv_bn_prelu_conv = nn.Sequential(
            nn.Conv2d(num_input, num_input, kernel_size,
                      groups=num_input, padding=padding, bias=False),
            nn.BatchNorm2d(num_input, eps=0.001),
            nn.PReLU(num_input),
            nn.Conv2d(num_input, num_input, 1, bias=False),
            nn.BatchNorm2d(num_input, eps=0.001),
        )

    def forward(self, x):
        if self.pooling > 1:
            x = self.down_sample(x)

        x = self.conv_bn_prelu_conv(x)

        if self.pooling > 1:
            x = self.up_sample(x)

        return x


class S2Module(nn.Module):
    def __init__(self, num_input, num_output, config=((3, 1), (5, 1)), skip=False):
        super().__init__()

        self.num_groups = len(config)

        channel_per_group = num_output//self.num_groups

        self.conv = nn.Conv2d(num_input, channel_per_group,
                              1, 1, groups=self.num_groups, bias=False)

        self.s2_blocks = nn.ModuleList()
        for i, config in enumerate(config):
            self.s2_blocks.append(S2Block(channel_per_group, config))

        self.bn = nn.BatchNorm2d(num_output, eps=0.001)
        self.prelu = nn.PReLU(num_output)

        self.skip = skip

    def forward(self, x):
        skip = x
        x = self.conv(x)

        x = channel_shuffle(x, self.num_groups)

        result = []
        for s2_block in self.s2_blocks:
            result.append(s2_block(x))

        x = torch.cat(result, 1)

        if self.skip:
            x = x + skip

        x = self.bn(x)
        x = self.prelu(x)

        return x


class SINet(nn.Module):
    def __init__(self, train_encoder_only=False):
        super().__init__()
        self.cbr_1 = ConvBnPRelu(3, 12, 3, 2)

        self.ds_conv_se_1 = DSConvSE(12, 16, 3, 2, reduction=1)

        self.sb_modules_1 = nn.ModuleList()

        for i, cfg in enumerate((((3, 1), (5, 1)), ((3, 1), (3, 1)))):
            if i == 0:
                self.sb_modules_1.append(
                    S2Module(16, 48, config=cfg, skip=False))
            else:
                self.sb_modules_1.append(
                    S2Module(48, 48, config=cfg, skip=True))

        self.br_1 = BnPRelu(48 + 16)

        self.ds_conv_se_2 = DSConvSE(64, 48, 3, 2)

        config = (
            ((3, 1), (5, 1)),
            ((3, 1), (3, 1)),
            ((5, 1), (3, 2)),
            ((5, 2), (3, 4)),

            ((3, 1), (3, 1)),
            ((5, 1), (5, 1)),
            ((3, 2), (3, 4)),
            ((3, 1), (5, 2)),
        )

        self.sb_modules_2 = nn.ModuleList()
        for i, cfg in enumerate(config):
            if i == 0:
                self.sb_modules_2.append(
                    S2Module(48, 96, config=cfg, skip=False))
            else:
                self.sb_modules_2.append(
                    S2Module(96, 96, config=cfg, skip=True))

        self.br_3 = BnPRelu(96 + 48)

        self.low_level_conv = nn.Conv2d(48, 1, 1, bias=False)

        self.conv_3 = nn.Conv2d(144, 1, 1, bias=False)

        self.classifier = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        self.train_encoder_only = train_encoder_only

    def forward(self, x):
        x = self.cbr_1(x)

        x = self.ds_conv_se_1(x)
        skip = x

        for sb in self.sb_modules_1:
            x = sb(x)

        if not self.train_encoder_only:
            low_level = self.low_level_conv(x)

        x = torch.cat((x, skip), 1)

        x = self.br_1(x)

        x = self.ds_conv_se_2(x)

        skip = x

        for sb in self.sb_modules_2:
            x = sb(x)

        x = torch.cat((x, skip), dim=1)

        x = self.br_3(x)

        x = self.conv_3(x)

        if not self.train_encoder_only:
            x = F.interpolate(x, scale_factor=2,
                              mode='bilinear', align_corners=True)

            c = torch.argmax(F.softmax(x, dim=1), dim=1).unsqueeze(1)

            info_block = 1 - c

            blocked_low_level = info_block * low_level

            x = x + blocked_low_level

        if not self.train_encoder_only:
            x = F.interpolate(x, scale_factor=2,
                              mode='bilinear', align_corners=True)

            x = self.classifier(x)

            x = F.interpolate(x, scale_factor=2,
                              mode='bilinear', align_corners=True)

        return x
