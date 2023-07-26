import numpy as np
import cv2

from pathlib import Path
from pyquaternion import Quaternion

from transforms3d.euler import euler2quat, euler2mat

import torch, math


INTERPOLATION = cv2.LINE_8


def get_split(split, dataset_name):
    split_dir = Path(__file__).parent / 'splits' / dataset_name
    split_path = split_dir / f'{split}.txt'

    return split_path.read_text().strip().split('\n')


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ])


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


def get_pose(rotation, translation, inv=False, flat=False):
    if flat: 
        yaw = Quaternion(rotation).yaw_pitch_roll[0]    
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix
    t = np.array(translation, dtype=np.float32)
    return get_transformation_matrix(R, t, inv=inv)

def encode(x):
    """
    (h, w, c) np.uint8 {0, 255}
    """
    n = x.shape[2]

    # assert n < 16
    assert x.ndim == 3
    assert x.dtype == np.uint8
    assert all(x in [0, 255] for x in np.unique(x))

    shift = np.arange(n, dtype=np.int32)[None, None]

    binary = (x > 0)
    binary = (binary << shift).sum(-1)
    binary = binary.astype(np.int32)

    return binary


def decode(img, n):
    """
    returns (h, w, n) np.int32 {0, 1}
    """
    shift = np.arange(n, dtype=np.int32)[None, None]

    x = np.array(img)[..., None]
    x = (x >> shift) & 1

    return x


if __name__ == '__main__':
    from PIL import Image

    n = 12

    x = np.random.rand(64, 64, n)
    x = 255 * (x > 0.5).astype(np.uint8)

    x_encoded = encode(x)
    x_img = Image.fromarray(x_encoded)
    x_img.save('tmp.png')
    x_loaded = Image.open('tmp.png')
    x_decoded = 255 * decode(x_loaded, 12)
    x_decoded = x_decoded[..., :n]

    print(abs(x_decoded - x).max())

def mask(img, target):
    m = np.all(img == target, axis=2).astype(int)
    return m

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])
    
def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

# converts rotation from euler angles to rotation matrix
def get_camera_info(rotation, translation):
    roll = math.radians(rotation[2] - 90)
    pitch = -math.radians(rotation[1])
    yaw = -math.radians(rotation[0])
    rotation_matrix = euler2mat(roll, pitch, yaw)

    return torch.tensor(rotation_matrix), torch.tensor(translation)

def get_intrinsics(sensor_options):
    calibration = np.identity(3)
    calibration[0, 2] = sensor_options['image_size_x'] / 2.0
    calibration[1, 2] = sensor_options['image_size_y'] / 2.0
    calibration[0, 0] = calibration[1, 1] = sensor_options['image_size_x'] / (
            2.0 * np.tan(sensor_options['fov'] * np.pi / 360.0))
    return torch.tensor(calibration)