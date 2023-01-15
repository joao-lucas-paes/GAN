import torch
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, in_layer) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_layer, 64, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Sigmoid(),
            nn.Dropout2d(0.2, inplace=False)
        )

    def forward(self, input):
        return self.main(input).view(-1)