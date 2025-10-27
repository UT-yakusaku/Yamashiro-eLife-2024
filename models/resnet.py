import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
 
class ResNet_1D_Custom_Multihead(nn.Module):
    def __init__(self, layers, chs = 32, num_classes=2):
        super(ResNet_1D_Custom_Multihead, self).__init__()
        self.chs = chs

        # left
        self.conv1_L1 = nn.Sequential(
            nn.Conv1d(self.chs, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

        self.conv1_L2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(8, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

        # right
        self.conv1_R = nn.Sequential(
            nn.Conv1d(self.chs, 8, kernel_size=41, stride=8, padding=20),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.inchannel = 16

        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(ResidualBlock, 16, layers[0], stride=2)
        self.layer1 = self._make_layer(ResidualBlock, 16, layers[1], stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc_head1 = nn.Linear(912, 1) # TODO: check the input size
        self.fc_head2 = nn.Linear(912, 1) # TODO: check the input size

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inchannel != channel:
            downsample = nn.Sequential(
                nn.Conv1d(self.inchannel, channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(channel),
            )
        layers = []
        layers.append(block(self.inchannel, channel, stride, downsample))
        self.inchannel = channel
        for i in range(1, blocks):
            layers.append(block(self.inchannel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_L = self.conv1_L1(x)
        x_L = self.conv1_L2(x_L)

        x_R = self.conv1_R(x)
        # concatenate
        x = torch.cat((x_L, x_R), 1)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        o1 = self.fc_head1(x)
        o2 = self.fc_head2(x)

        output = torch.cat((o1, o2), 1)

        return output