import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from pathlib import Path


class SiameseNetworkDataset(Dataset):
    """
    PyTorch Dataset Class which gets data from a csv file
    which was initialized within the processing step (./notebooks/DataPreparation_V2.ipynb)
    """

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.df.loc[index]

        # from pathlib import Path

        # full_path = str(Path.cwd())
        # TODO parameterize Linux, Mac paths ...
        # since I am an idiot and I used absolut paths I have to make weird changes in path
        img_0 = Image.open('/home/hpc/iwi5/iwi5012h/dev/jigsaw-puzzle-solver' + sample['0'] + '.jpg')
        img_1 = Image.open('/home/hpc/iwi5/iwi5012h/dev/jigsaw-puzzle-solver' + sample['1'] + '.jpg')

        #img_0 = Image.open('/Users/beantown/PycharmProjects/jigsaw-puzzle-solver' + sample['0'] + '.jpg')
        #img_1 = Image.open('/Users/beantown/PycharmProjects/jigsaw-puzzle-solver' + sample['1'] + '.jpg')

        y = torch.from_numpy(np.array(sample.y, dtype=np.float32))


        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        return img_0, img_1, y

    def __len__(self):
        length = self.df.shape[0]
        return length
