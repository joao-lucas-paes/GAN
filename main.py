from model.Gan import Gan
from albumentations.pytorch import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch

IMAGE_HEIGHT = 500
IMAGE_WIDTH  = 500

train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=35, p=.40),
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.1),
    A.ToFloat(max_value=255),
    ToTensorV2(),
],)

def getImgs():
    with open('./data/train_images.txt', 'r') as f:
        img_train = [el[:-1] for el in f.readlines()]

    with open('./data/val_images.txt', 'r' ) as f:
        img_val = [el[:-1] for el in f.readlines()]

    with open('./data/train_masks.txt', 'r') as f:
        mask_train= [el[:-1] for el in f.readlines()]

    with open('./data/val_masks.txt', 'r') as f:
        mask_val= [el[:-1] for el in f.readlines()]
    
    return [img_train, mask_train, img_val, mask_val]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_test = Gan(500, 4)

model_test.load()

# noise = torch.randn(4, 4, 500, 500).to(DEVICE)

# imgs = model_test.generator(noise).cpu().detach().numpy()[:, :3, :, :]

# for i in range(imgs.shape[0]):
#     imgs[i, 0] = (imgs[i, 0, :, :] - imgs[i, 0, :, :].min()) / (imgs[i, 0, :, :].max() - imgs[i, 0, :, :].min())
#     imgs[i, 1] = (imgs[i, 1, :, :] - imgs[i, 1, :, :].min()) / (imgs[i, 1, :, :].max() - imgs[i, 1, :, :].min())
#     imgs[i, 2] = (imgs[i, 2, :, :] - imgs[i, 2, :, :].min()) / (imgs[i, 2, :, :].max() - imgs[i, 2, :, :].min())

#     cop = imgs[i].copy()

#     cop = cop.reshape((500, 500, 3))

#     plt.imshow(cop)
#     plt.show()

[img_t, mask_t, _, __] = getImgs()

model_test.train(img_t, mask_t, train_transform, 300, 2)