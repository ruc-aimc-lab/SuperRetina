import torch
import scipy.stats as st
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from common.common_util import pre_processing
from common.train_util import get_gaussian_kernel
import time

class RetinaDataset(Dataset):
    def __init__(self, data_path, split_file='eccv22_train.txt',
                 is_train=True, data_shape=(768, 768), auxiliary=None):
        self.data = []

        self.image_path = os.path.join(data_path, 'ImageData')
        self.label_path = os.path.join(data_path, 'Annotations')
        self.split_file = os.path.join(data_path, 'ImageSets', split_file)

        self.enhancement_sequential = iaa.Sequential([
            iaa.Multiply((1.0, 1.2)),  # change brightness, doesn't affect keypoints
            iaa.Sometimes(
                0.2,
                iaa.GaussianBlur(sigma=(0, 6))
            ),
            iaa.Sometimes(
                0.2,
                iaa.LinearContrast((0.75, 1.2))
            ),
        ], random_order=True)

        self.is_train = is_train
        self.model_image_height, self.model_image_width = data_shape[0], data_shape[1]

        with open(self.split_file) as f:
            files = f.readlines()

        for file in files:
            file = file.strip()
            try:
                image, label = file.split(', ')
                image = os.path.join(self.image_path, image)
                label = os.path.join(self.label_path, label)
                assert os.path.exists(image) and os.path.exists(label)
                self.data.append((image, label))
            except Exception:
                pass
        if auxiliary is not None and os.path.exists(auxiliary):
            print('-'*10+f"Load Lab {'train' if is_train else 'eval'} data and auxiliary data without labels"+'-'*10)
            auxiliaries_tmp = os.listdir(auxiliary)
            auxiliaries = []
            auxiliaries_tmp = [(os.path.join(auxiliary, a), None) for a in auxiliaries_tmp]
            for a, b in auxiliaries_tmp:
                if a.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    auxiliaries.append((a, b))
            self.data = self.data + auxiliaries
        else:
            print('-' * 10 + f"Load Lab {'train' if is_train else 'eval'} data, and there is no auxiliary data" + '-' * 10)
        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_with_label = False

        image_path, label_path = self.data[index]
        label_name = '[None]'

        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        image = image[:, :, 1]

        image = pre_processing(image)
        # if self.is_train:
        #     image = self.enhancement_sequential(image=image)
        image = Image.fromarray(image)
        image_tensor = self.transforms(image)

        if label_path is not None:
            label_name = os.path.split(label_path)[-1]
            keypoint_position = np.loadtxt(label_path)  # (2, n): (x, y).T
            keypoint_position[:, 0] *= image_tensor.shape[-1]
            keypoint_position[:, 1] *= image_tensor.shape[-2]

            tensor_position = torch.zeros([self.model_image_height, self.model_image_width])

            tensor_position[keypoint_position[:, 1], keypoint_position[:, 0]] = 1
            tensor_position = tensor_position.unsqueeze(0)
            input_with_label = True

            return image_tensor, input_with_label, tensor_position, label_name

        # without labels, only train descriptor
        tensor_position = torch.empty(image_tensor.shape)

        return image_tensor, input_with_label, tensor_position, label_name



