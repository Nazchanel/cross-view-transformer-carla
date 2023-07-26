import sys
import os

# Add the top-level directory to the PYTHONPATH
top_level_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_level_dir)

import torch, json, math
from PIL import Image
import numpy as np
import torchvision
from cross_view_transformer.data.common import get_transformation_matrix
from .transforms import SaveCarlaDataTransform
from .common import get_camera_info, get_intrinsics

normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                ))

class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg, record_path):
        self.data_cfg = data_cfg
        self.record_path = record_path
        self.vehicles = len(os.listdir(os.path.join(self.record_path, 'agents'))) # total agents (we only have 1)
        self.ticks = len(os.listdir(os.path.join(self.record_path, 'agents/0/back_camera'))) # total frames per record
        
        with open(os.path.join(self.record_path, 'agents/0/sensors.json'), 'r') as f:
            self.sensors_info = json.load(f)['sensors']
        
        self.cameras = ["left_front_camera", "front_camera", "right_front_camera", "left_back_camera", "back_camera", "right_back_camera"]
        
        self.camera_imgs = {i : self.read_image_folder(os.path.join(self.record_path, 'agents/0', i)) for i in self.cameras}
        self.bev_imgs = self.read_image_folder(os.path.join(self.record_path, 'agents/0/birds_view_semantic_camera'))
        
        self.classes = ['vehicle', 'drivable_area', 'lane_markings']
        
        self.data_transform = SaveCarlaDataTransform() # used to convert the files into json format
        
    def read_image_folder(self, image_folder_path):
        imgs = []
        for i in os.listdir(image_folder_path):
            imgs.append(os.path.join(image_folder_path, i))
        return imgs

    
    def __len__(self):
        return self.vehicles*self.ticks
    
    def __getitem__(self, idx):
    
        imgs = []
        intrins = []
        extrins = []

        bev = self.bev_imgs[idx] # image path of bev
        
        cam_idx = torch.arange(0, 6)

        lidar_rotation = self.sensors_info['lidar']['transform']['rotation']
        lidar_translation = self.sensors_info['lidar']['transform']['location']

        # convert rotation to matrix form
        lidar_rotation, lidar_translation = get_camera_info(lidar_rotation, lidar_translation)
        pose = get_transformation_matrix(lidar_rotation, lidar_translation)
        
        for sensor_name, sensor_info in self.sensors_info.items():
            if sensor_name in self.cameras:
                image_path = self.camera_imgs[sensor_name][idx]

                tran = sensor_info["transform"]["location"]
                rot = sensor_info["transform"]["rotation"]
                sensor_options = sensor_info["sensor_options"]

                # converts rotation from euler angles to rotation matrix
                rot, tran = get_camera_info(rot, tran)
                
                intrin = get_intrinsics(sensor_options)
                extrin = torch.cat((rot, tran.unsqueeze(1)), dim=1)
                extrin = torch.cat((extrin, torch.tensor([0, 0, 0, 1]).view(1, 4)), dim=0)
                
                imgs.append(image_path)
                intrins.append(intrin)
                extrins.append(extrin)
        
        data_sample = {'cam_idx': cam_idx,
                    'image': imgs, 
                    'intrinsics': torch.stack(intrins).float(), 
                    'extrinsics': torch.stack(extrins).float(), 
                    'bev': bev, 
                    'pose': pose.tolist()
                    }
        
        data_sample = self.data_transform.transform(data_sample)
        return data_sample
        
        
