import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, in_channels=64, out_channels=4, img_size=256):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, 1, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, 1, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 3, 1, 0, bias=False),
            nn.Tanh()
        )
        self.img_size = img_size
        self.out_channels = out_channels
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(input.shape[0], self.out_channels, self.img_size, self.img_size) 
        return output