import pandas as pd
from cv2 import imread
from torch.utils.data import Dataset


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
        print(sample[0])
        print(sample[1])
        img_0 = imread(sample[0] + '.jpg')
        img_1 = imread(sample[1] + '.jpg')
        y = sample.y

        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        return {'img_0': img_0, 'img_1': img_1, 'label': y}

    def __len__(self):
        length = self.df.shape[0]
        print(length)
        return length
