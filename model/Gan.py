from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model.Generator_GAN_DCGAN import Generator
from model.Discriminator_GAN_DCGAN import Discriminator
from model.CustomDataset import CartoonDataset
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_VALUE = 0.01

class Gan():
    def __init__(self, in_channels=3, in_bands=3, lr=1e-4) -> None:
        self.generator = Generator(in_bands, in_bands, 500).to(DEVICE)
        self.discrimator = Discriminator(in_bands).to(DEVICE)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.discrimator.parameters(), lr=lr)

        self.d_criterion = nn.BCELoss().to(DEVICE)
        self.g_criterion = nn.BCELoss().to(DEVICE)
        self.wasserstein = optim.RMSprop(self.discrimator.parameters(), lr=lr)

        self.in_channels = in_channels
        self.in_bands = in_bands

    @staticmethod
    def generate_labels(size:int) -> List[torch.Tensor]:
        return [
            torch.ones(size).to(DEVICE), 
            torch.zeros(size).to(DEVICE)
        ]
    
    @staticmethod
    def printStatus(d_values, g_values, epoch_num) -> None:
        """ Show status train """
        print("\r=======================================")
        print(f"Epoch {epoch_num}:")
        print("d_loss: \t | \t g_loss:")
        print(f"{d_values.item():.8f} \t | \t {g_values.item():.8f}")
        print("=======================================")
    
    def get_tensors_img(self, index, batch_size, data_loader) -> torch.Tensor:
        output = torch.empty(batch_size, self.in_bands, self.in_channels, self.in_channels)
        
        for i in range(batch_size):
            index_in = index + i
            if(index_in < len(data_loader)):
                output[i] = data_loader[index_in][0]
            else:
                break

        return output


    def train(self, img_path:str, mask_path:str, transform=None, epochs:int=200, batch_size:int=100) -> None:
        """ train - process abstracted """
        [r_labels, f_labels] = self.generate_labels(batch_size)
        noise = torch.randn(batch_size, self.in_bands, 500, 500).to(DEVICE)
        data_loader = CartoonDataset(transform)

        for epoch in range(epochs):
            d_loss, g_loss = None, None
            for index in range(0, len(data_loader), batch_size):
                real_img = self.get_tensors_img(index, batch_size, data_loader).to(DEVICE)
                fake_img = self.generator(noise).to(DEVICE)
                
                d_loss = self.discriminator_train(real_img, r_labels, fake_img, f_labels)
                g_loss = self.generator_train(r_labels, fake_img)

                Gan.print_progress(index, len(data_loader))
            
            self.save(epoch)
            self.printStatus(d_loss, g_loss, epoch)
    
    @staticmethod
    def print_progress(index, length):
        perc = 100 * index / length
        in_bars = int(perc/2.5)
        bar = "█" * in_bars
        invisible_side = " " * (40 - in_bars)
        print(f"Progress: {perc:.2f}% |{bar}{invisible_side}|", end="\r")

    def clipping_fn(self):
        for p in self.discrimator.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
        self.wasserstein.step()

    def discriminator_train(self, r_img,  r_labels, f_img, f_labels) -> tuple:
        """ train the discriminator """
        self.d_optimizer.zero_grad()

        r_output = self.discrimator(r_img)
        r_loss = self.d_criterion(r_output, r_labels)

        f_output = self.discrimator(f_img.detach())
        f_loss = self.d_criterion(f_output, f_labels)

        loss = (r_loss + f_loss) / 2
        loss.backward()

        self.clipping_fn()
        self.d_optimizer.step()
        
        return loss
    
    def generator_train(self, r_label, f_img):
        """ train the generator """
        self.g_optimizer.zero_grad()

        output = self.discrimator(f_img)
        loss = self.g_criterion(output, r_label)
        loss.backward()

        self.g_optimizer.step()

        return loss

    def save(self, epoch):
        torch.save(self.discrimator.state_dict(), "disc.pth")
        torch.save(self.generator.state_dict(), "gen.pth")
        noise = torch.randn(1, self.in_bands, 500, 500)
        img = (self.normalized_out(noise)[0] * 255).astype('uint8')
        Image.fromarray(img, "RGBA").save(f"./out/gen_{epoch}.png")
    
    def load(self, path="./"):
        self.discrimator.load_state_dict(torch.load(path + "disc.pth"))
        self.generator.load_state_dict(torch.load(path + "gen.pth"))
    
    def normalize(self, input):
        in_shape = input.shape
        imgs = np.copy(input).reshape([in_shape[0], in_shape[2], in_shape[3], in_shape[1]])
        norm_rng = lambda rng : (rng - rng.min()) / (rng.max() - rng.min())
        
        for i in range(in_shape[0]):
            for j in range(in_shape[1]):
                imgs[i, :, :, j] = norm_rng(imgs[i, :, :, j])

        return imgs

    def normalized_out(self, noise):
        noise_device = noise.to(DEVICE)
        imgs = self.generator(noise_device).cpu().detach().numpy()
        return self.normalize(imgs)