import json
import torch
import os

from .transforms import LoadCarlaDataTransform

class CarlaDatasetGenerated(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, data_cfg, json_file='data.json', transform=LoadCarlaDataTransform):
        # this is a list of all the cameras associated with a specific scene
        json_dir = data_cfg.json_dir
        json_file = os.path.join(json_dir, json_file)
        with open(json_file, "r") as file:
            self.samples = json.load(file) 
        self.transform = transform(data_cfg.image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[str(idx)]

        if self.transform is not None:
            data = self.transform(data)

        return data
