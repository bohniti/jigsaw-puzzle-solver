import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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

        img_0 = Image.open(sample[0] + '.jpg')
        img_1 = Image.open(sample[1] + '.jpg')

        y = torch.from_numpy(np.array(sample.y, dtype=np.float32))


        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        return img_0, img_1, y

    def __len__(self):
        length = self.df.shape[0]
        return length
