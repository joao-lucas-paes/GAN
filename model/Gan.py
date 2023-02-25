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
    def __init__(self, in_channels=3, in_bands=3, lr=1e-5) -> None:
        self.generator = Generator(in_bands, in_bands, 512).to(DEVICE)
        self.discriminator = Discriminator(in_bands).to(DEVICE)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.d_criterion = nn.BCELoss().to(DEVICE)
        self.g_criterion = nn.MSELoss().to(DEVICE)
        self.wasserstein = optim.RMSprop(self.discriminator.parameters(), lr=lr)

        self.d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer, gamma=0.1, step_size=10)
        self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, gamma=0.1, step_size=20)

        self.in_channels = in_channels
        self.in_bands = in_bands

    @staticmethod
    def generate_labels(size:int) -> List[torch.Tensor]:
        return [
            torch.ones(size).to(DEVICE), 
            torch.zeros(size).to(DEVICE)
        ]
    
    def printStatus(self, d_values, g_values, epoch_num) -> None:
        """ Show status train """
        print("\r=======================================")
        print(f"Epoch {epoch_num}:")
        print("d_loss: \t | \t g_loss:")
        print(f"{d_values.item():.8f} \t | \t {g_values.item():.8f}")
        print("d_lr: \t | \t g_lr:")
        print(f"{self.d_optimizer.param_groups[0]['lr']} \t | \t {self.g_optimizer.param_groups[0]['lr']}")
        print("=======================================")

    def printStatusD(self, d_values, epoch_num) -> None:
        """ Show status train """
        print("\r=======================================")
        print(f"Epoch {epoch_num}:")
        print("d_loss: \t | \t g_loss:")
        print(f"{d_values.item():.8f} \t | \t {'não treinado'}")
        print("d_lr: \t | \t g_lr:")
        print(f"{self.d_optimizer.param_groups[0]['lr']} \t | \t {self.g_optimizer.param_groups[0]['lr']}")
        print("=======================================")
    
    def get_tensors_img(self, index, batch_size, data_loader) -> torch.Tensor:
        output = torch.empty(batch_size, self.in_bands, self.in_channels, self.in_channels)
        
        for i in range(batch_size):
            index_in = index + i
            if(index_in < len(data_loader)):
                output[i] = data_loader[index_in]
            else:
                break

        return output


    def train(self, img_path:str, mask_path:str, transform=None, epochs:int=200, batch_size:int=100) -> None:
        """ train - process abstracted """
        [r_labels, f_labels] = self.generate_labels(batch_size)
        noise = torch.randn(batch_size, self.in_bands, 32, 32).to(DEVICE)
        data_loader = CartoonDataset(transform)

        for epoch in range(epochs):
            d_loss, g_loss = None, None
            for index in range(0, len(data_loader), batch_size):
                real_img = self.get_tensors_img(index, batch_size, data_loader).to(DEVICE)
                
                d_loss = self.discriminator_train(real_img, r_labels, f_labels, noise)
                if (epoch % 2 != 0):
                    g_loss = self.generator_train(r_labels, noise)

                Gan.print_progress(index, len(data_loader))

            with torch.no_grad():
                self.d_scheduler.step()
                self.g_scheduler.step()
            
                self.save(epoch)
                if (epoch % 2 == 0):
                    self.printStatusD(d_loss, epoch)
                else:
                    self.printStatus(d_loss, g_loss, epoch)
    
    @staticmethod
    def print_progress(index, length):
        perc = 100 * index / length
        in_bars = int(perc/2.5)
        bar = "█" * in_bars
        invisible_side = " " * (40 - in_bars)
        print(f"Progress: {perc:.2f}% |{bar}{invisible_side}|", end="\r")

    def clipping_fn(self):
        for p in self.discriminator.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
        self.wasserstein.step()

    def discriminator_train(self, r_img,  r_labels, f_labels, noise) -> tuple:
        """ train the discriminator """
        f_img = self.generator(noise).to(DEVICE)

        self.d_optimizer.zero_grad()

        r_output = self.discriminator(r_img)
        r_loss = self.d_criterion(r_output, r_labels)

        f_output = self.discriminator(f_img)
        f_loss = self.d_criterion(f_output, f_labels)

        loss = (r_loss + f_loss) / 2
        loss.backward()

        self.clipping_fn()
        self.d_optimizer.step()
        
        return loss
    
    def generator_train(self, r_label, noise):
        """ train the generator """
        f_img = self.generator(noise).to(DEVICE)

        self.g_optimizer.zero_grad()

        output = self.discriminator(f_img)
        loss = self.g_criterion(output, r_label)
        loss.backward()

        self.g_optimizer.step()

        return loss

    def save(self, epoch):
        torch.save(self.discriminator.state_dict(), "disc.pth")
        torch.save(self.generator.state_dict(), "gen.pth")
        noise = torch.randn(1, self.in_bands, 32, 32)
        img = self.imgs_out(noise)[0]
        Image.fromarray(img, "RGBA").save(f"./out/gen_{epoch}.png")
    
    def load(self, path="./"):
        self.discriminator.load_state_dict(torch.load(path + "disc.pth"))
        self.generator.load_state_dict(torch.load(path + "gen.pth"))

    def imgs_out(self, noise):
        noise_device = noise.to(DEVICE)
        imgs = self.generator(noise_device).cpu().detach().clamp_(0, 1).numpy().transpose(0, 2, 3, 1).astype(np.float32)
        return (imgs * 255).astype(np.uint8)