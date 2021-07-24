from sys import platform

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from torchvision import transforms

from SiameseNetworkDataset import SiameseNetworkDataset
from models.SiameseNetwork_V1 import SiameseNetwork
from utils.utils import load_config


def main(config_file_name):
    config = load_config(config_file_name, show=True, tune_config=False)
    transform = transforms.Compose([
        transforms.CenterCrop(config['center_crop']),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset1 = SiameseNetworkDataset(
        csv_file=config['train_file'], raw_img_path=config['raw_img_path'],
        transform=transform)

    dataset2 = SiameseNetworkDataset(
        csv_file=config['val_file'], raw_img_path=config['raw_img_path'],
        transform=transform)

    train_dataloader = DataLoader(dataset1, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                  drop_last=True)
    val_dataloader = DataLoader(dataset2, batch_size=config['batch_size'], num_workers=config['num_workers'])

    model = SiameseNetwork()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=config['save_dir'])
    trainer = pl.Trainer(min_epochs=config['min_epochs'], max_epochs=config['max_epochs'], logger=tb_logger,
                         gpus=config['gpus'], default_root_dir=config['default_root_dir'])
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":
        config = 'config'
    elif platform == "darwin":
        config = 'config_local'
    elif platform == "win32":
        raise NotImplementedError

    main(config)
