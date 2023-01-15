from model.Gan import Gan
from albumentations.pytorch import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 256

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

model_test = Gan(256, 4)

[img_t, mask_t, _, __] = getImgs()

model_test.train(img_t, mask_t, train_transform, 1000, 2)