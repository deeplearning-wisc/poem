import torch.nn as nn
import torch

class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=80):

        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1024, bias=True),
            nn.GroupNorm(num_channels=1024, num_groups=32),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.cls= nn.Linear(1024, num_classes,bias=True)

    def forward(self, x):
        # bp()
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        return x
