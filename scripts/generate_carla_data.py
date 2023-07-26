
import hydra
from pathlib import Path
import os
import json
from cross_view_transformer.common import setup_config
from cross_view_transformer.data.carla_dataset import CarlaDataset

def setup(cfg):
    # Don't change these
    cfg.data.augment = 'none'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = True
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False

    # Uncomment to debug errors hidden by multiprocessing
    # cfg.loader.num_workers = 0
    # cfg.loader.prefetch_factor = 2
    # cfg.loader.persistent_workers = False

CONFIG_PATH = os.path.join(os.getcwd(), 'config')
CONFIG_NAME = 'config.yaml'

def create_regular_data(cfg):
    samples = []
    for town in cfg.data.towns:
        current_town_dir = os.path.join(cfg.data.dataset_dir, town)
        dataset = CarlaDataset(cfg.data, current_town_dir)
        for i in range(len(dataset)):
            samples.append(dataset.__getitem__(i))
        
    names = {i: sample for i, sample in enumerate(samples)}
    with open(os.path.join(cfg.data.json_dir, "data.json"), "w") as file_path:
        json.dump(names, file_path)

def create_ood_data(cfg):
    ood_town_dir = os.path.join(cfg.data.dataset_dir, cfg.data.ood_town)
    ood_dataset = CarlaDataset(cfg.data, ood_town_dir)
    ood_samples = []
    for i in range(len(ood_dataset)):
        ood_samples.append(ood_dataset.__getitem__(i))
    
    ood_names = {i: sample for i, sample in enumerate(ood_samples)}
    with open(os.path.join(cfg.data.json_dir, "ood_data.json"), "w") as file_path:
        json.dump(ood_names, file_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg, setup)
    os.makedirs(cfg.data.json_dir, exist_ok=True)

    create_regular_data(cfg)
    create_ood_data(cfg)

    
if __name__ == '__main__':
    main()