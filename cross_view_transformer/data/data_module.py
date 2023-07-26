from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split

from . import get_dataset_module_by_name


class NuscenesDataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        # originally dataset was set to 'nuscenes_dataset_generated' in the config files
        self.get_data = get_dataset_module_by_name(dataset).get_data 

        self.data_config = data_config
        self.loader_config = loader_config

    def get_split(self, split, loader=True, shuffle=False):
        datasets = self.get_data(split=split, **self.data_config)

        if not loader:
            return datasets

        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_config)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=True):
        return self.get_split('train', loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', loader=True, shuffle=shuffle)
    
class CarlaDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, loader_cfg):
        from cross_view_transformer.data.carla_dataset_generated import CarlaDatasetGenerated
        super().__init__()
        self.dataset = CarlaDatasetGenerated(data_cfg)
        
        self.train_dataset, self.val_dataset = self.split_dataset()
        # self.train = torch.utils.data.DataLoader(train_dataset, **loader_cfg)
        self.train = torch.utils.data.DataLoader(self.train_dataset, **loader_cfg)
        self.val = torch.utils.data.DataLoader(self.val_dataset, **loader_cfg)
        
        self.test_dataset = CarlaDatasetGenerated(data_cfg, json_file='ood_data.json')
        self.test = torch.utils.data.DataLoader(self.test_dataset, **loader_cfg)
    
    def split_dataset(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, test_size])
        return train_dataset, val_dataset

    def train_dataloader(self):
        return self.train
    
    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test