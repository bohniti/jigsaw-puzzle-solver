import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms

from langdon.features.core import SiameseDataset
from langdon.models.siamese_resnet.core import SiameseNetwork
from langdon.utils.core import load_config, get_config_path


def init_step():
    config_file_name = get_config_path()
    config = load_config(config_file_name, show=True, tune_config=False)
    transform = transforms.Compose([
        transforms.CenterCrop(config['center_crop']),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    model = SiameseNetwork(batch_size=config['batch_size'],
                           learning_rate=config['learning_rate'],
                           margin=config['margin'],
                           partial_conf=config['partial_conf'],
                           center_crop=config['center_crop'],
                           linear_input=config['linear_input'])

    return config, transform, model


def load_step(config, transform):
    dataset1 = SiameseDataset(
        csv_file=config['train_file'],
        raw_img_path=config['raw_img_path'],
        transform=transform)
    dataset2 = SiameseDataset(
        csv_file=config['val_file'],
        raw_img_path=config['raw_img_path'],
        transform=transform)

    train_dataloader = torch.utils.data.DataLoader(dataset1,
                                                   batch_size=config['batch_size'],
                                                   num_workers=config['num_workers'],
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(dataset2,
                                                 batch_size=config['batch_size'],
                                                 num_workers=config['num_workers'],
                                                 drop_last=True)

    if config['stage'] == 'train':
        return train_dataloader, val_dataloader
    else:
        dataset3 = SiameseDataset(
            csv_file=config['test_file'],
            raw_img_path=config['raw_img_path'],
            transform=transform)
        testt_dataloader = torch.utils.data.DataLoader(dataset3,
                                                       batch_size=config['batch_size'],
                                                       num_workers=config['num_workers'],
                                                       drop_last=True)

        return train_dataloader, val_dataloader, testt_dataloader


def log_step(config):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config['save_dir'])
    tb_logger.log_hyperparams({'min_epohcs': config['min_epochs'],
                               'max_epochs': config['max_epochs'],
                               'learning_rate': config['learning_rate'],
                               'batch_size': config['batch_size'],
                               'margin': config['margin'],
                               'center_crop': config['center_crop']})
    return tb_logger


def train_step(config, model, train_dataloader, val_dataloader, tb_logger, test_dataloader=None):
    if config['stage'] == 'train':
        trainer = pl.Trainer(min_epochs=config['min_epochs'],
                             max_epochs=config['max_epochs'],
                             logger=tb_logger,
                             gpus=config['gpus'],
                             default_root_dir=config['default_root_dir'],
                             progress_bar_refresh_rate=config['progress_bar_refresh_rate'])
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer = pl.Trainer(min_epochs=config['min_epochs'],
                             max_epochs=config['max_epochs'],
                             logger=tb_logger,
                             gpus=config['gpus'],
                             default_root_dir=config['default_root_dir'],
                             progress_bar_refresh_rate=config['progress_bar_refresh_rate'],
                             )
        trainer.test(model, ckpt_path=config['checkpoint_path'], test_dataloaders=test_dataloader)
