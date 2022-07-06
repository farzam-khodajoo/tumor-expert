import importlib
import numpy as np
import torch
from torch.nn import Module


class BackgroundCrop(Module):
    """crop redundant background voxels (with voxel value zero)"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def bbox(image):
        """get bounding box of non-zero region"""
        rows = np.any(image, axis=1)
        cols = np.any(image, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return image[ymin:ymax+1, xmin:xmax+1]

    def forward(self, image_batch):
        if torch.is_tensor(image_batch):
            image_batch = image_batch.numpy()

        if len(image_batch.shape) == 2:
            return torch.tensor(self.bbox(image_batch))
        else:
            return torch.tensor(np.stack([self.bbox(image) for image in image_batch]))


class ModelGenerator:

    @staticmethod
    def number_of_features_per_level(init_channel_number, num_levels):
        return [init_channel_number * 2 ** k for k in range(num_levels)]

    @staticmethod
    def get_class(class_name, modules):
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz
        raise RuntimeError(f'Unsupported dataset class: {class_name}')