import torch.nn as nn
import torch.nn.functional as F

class DnCNN(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()

        kernel_size = 3
        padding = 1

        def conv_block(kernel_size, padding, n_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
                nn.ReLU(inplace=True)
            )

        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(conv_block(kernel_size, padding, n_channels))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        r = self.model(x)
        out = x-r
        return out

    def get_bn_before_relu(self):
        print(self.model)
        bn1 = self.model[5][-2]
        bn2 = self.model[7][-2]
        bn3 = self.model[17][-2]
        bn4 = self.model[19][-2]

        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):

        return [64, 64, 64, 64]

    def extract_feature(self, x, preReLU=False):

        feat1 = self.model[4][:-1](self.model[0:4](x))
        feat2 = self.model[6][:-1](self.model[5:6](F.relu(feat1)))
        feat3 = self.model[17][:-1](self.model[7:17](F.relu(feat2)))
        feat4 = self.model[19][:-1](self.model[18:19](F.relu(feat3)))

        out = self.model[20](F.relu(feat4))
        out = x-out

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        return [feat1, feat2, feat3, feat4], out