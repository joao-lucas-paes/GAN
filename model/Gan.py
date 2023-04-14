import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

import matplotlib.pyplot as plt 

class Gan:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_tensors_img(self, index, batch_size, data_loader) -> torch.Tensor:
        output = torch.empty(batch_size, 3, 64, 64)
        
        for i in range(batch_size):
            index_in = index + i
            if(index_in < len(data_loader)):
                output[i] = data_loader[index_in]
            else:
                break

        return output
        
    def train(self, dataloader, num_epochs, batch_size, lr, lr_milestones=[15, 30], gamma=0.5):
        criterion = nn.BCELoss()
        
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=lr_milestones, gamma=gamma)
        schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=lr_milestones, gamma=gamma)
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        fixed_noise = torch.randn(batch_size, 3, 4, 4, device=self.device)
        ds_size = len(dataloader)
        
        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()

            lossG = lossD_fake = lossD_real = 0

            output = [0]

            for i in range(0, ds_size, batch_size):
                data = self.get_tensors_img(i, batch_size, dataloader)
                
                real_images = data.to(self.device)
                real_labels = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
                fake_labels = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)
                noise = torch.randn(batch_size, 3, 4, 4, device=self.device)
                fake_images = self.generator(noise)

                
                if epoch % 1 == 0:
                    optimizerD.zero_grad()
                    output = self.discriminator(real_images)
                    lossD_real = criterion(output, real_labels)
                    lossD_real.backward()

                    output = self.discriminator(fake_images.detach())
                    lossD_fake = criterion(output, fake_labels)
                    lossD_fake.backward()
                    optimizerD.step()

                
                optimizerG.zero_grad()
                output = self.discriminator(fake_images)
                lossG = criterion(output, real_labels)
                lossG.backward()
                optimizerG.step()

                coef_perc = 2.5
                perc = (100*i/ds_size)
                squad_qtd = int(perc/coef_perc)
                squad = squad_qtd * "█"
                clear = (int(100/coef_perc) - squad_qtd) * " "
                print(f"Progress {perc:.2f}%:|{squad}{clear}|", end="\r")

            schedulerD.step()
            schedulerG.step()
            
            with torch.no_grad():
                fake_images = self.generator(fixed_noise)
                save_image(fake_images, f"output/{epoch}.png", nrow=int(batch_size**0.5), normalize=True)
            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, D_real: {lossD_real.item():.4f}, D_fake: {lossD_fake.item():.4f}, G: {lossG.item():.4f}\r")
                print(output[0])
            else:
                print(f"Epoch: {epoch}, G: {lossG.item():.4f}\r")

            torch.save(self.generator.state_dict(), f"models/generator_epoch_{num_epochs}.pth")
            torch.save(self.discriminator.state_dict(), f"models/discriminator_epoch_{num_epochs}.pth")
