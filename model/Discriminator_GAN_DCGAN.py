import torch
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, in_layer) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_layer, 64, 7, 2, 1, bias=False), # convolução com 3 canais de entrada, 64 de saida e um kernel de 7x7
            nn.LeakyReLU(0.2, inplace=True), # evitar camada nula usando a variação Leaky da ReLU
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1,1, bias=False)
        )

    def forward(self, input):
        return self.main(input).view(-1)