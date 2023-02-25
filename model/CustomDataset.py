from os import walk
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import rasterio

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = f'{self.image_paths[index]}/{self.image_paths[index].split("/")[-1]}.npy'
        image = np.load(path).reshape((256, 256, 4))

        path = self.mask_paths[index]
        with rasterio.open(f'{path}/raster_labels.tif') as dataset:
            mask = dataset.read(1)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

PATH_IMAGE_CARTOON = "./archive/cartoonset10k"

class CartoonDataset(Dataset):
    def __init__(self, transform=None):
        self.image_paths = next(walk(PATH_IMAGE_CARTOON), (None, None, []))[2]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = f'{PATH_IMAGE_CARTOON}/{self.image_paths[index]}'
        im = Image.open(path)
        image = np.asarray(im)
        image = image.reshape((im.width, im.height, 4))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image