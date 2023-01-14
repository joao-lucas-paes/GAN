import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 2048, 7, 1, 0, bias=False), # usando convolução, com "in_channels" de entrada, 512 de saida e kernel de 4x4 com transposição
            nn.BatchNorm2d(2048), # normalização por batch sobre um input de 4 dimensões -> (y = ((x - E[x])/(Var[x] - e)**0.5) * eps + gamma)
            nn.ReLU(True), # função de ativação não linear
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, out_channels, 4, 2, 1, bias=False),
            nn.Tanh() # função tangente hiperbólica
        )

    def forward(self, input):
        return self.main(input)