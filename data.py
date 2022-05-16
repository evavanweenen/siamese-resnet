import os
import pandas as pd
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class ImageDataset(Dataset):
    def __init__(self, datadir, split, img_size=256, crop_size=224):
        self.datadir = datadir
        self.split = split

        # read train/test triplets
        self.triplets = pd.read_csv(f'{datadir}/{split}_triplets.txt', sep=' ', header=None, dtype=str)
        self.triplets.columns = ['A', 'B', 'C']

        self.transforms = Compose([ Resize(img_size), 
                                    CenterCrop(crop_size),
                                    ToTensor(), 
                                    Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    def get_image(self, file):
        img = Image.open(f"{self.datadir}/food/{file}.jpg")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return tuple(self.get_image(file) for file in self.triplets.loc[idx])