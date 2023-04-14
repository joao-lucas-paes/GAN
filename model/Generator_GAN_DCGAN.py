import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module): #Deep convolutional GAN (DCGAN -> modelo convolucional)
    def __init__(self, in_channels=4, out_channels=4, img_size=256):
        super(Generator, self).__init__()
        i = 10
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 2 ** (i - 1), kernel_size=5, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 1), 2 ** (i - 2), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 2), 2 ** (i - 3), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 3), 2 ** (i - 3), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 3), 2 ** (i - 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 4)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 4), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), 2 ** (i - 5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 ** (i - 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 ** (i - 5), out_channels, kernel_size=2, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.img_size = img_size
        self.out_channels = out_channels
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(input.shape[0], self.out_channels, self.img_size, self.img_size) 
        return output