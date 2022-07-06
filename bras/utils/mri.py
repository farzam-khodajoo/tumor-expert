from typing import Union
from pathlib import Path
import nibabel as nib
import torch

class MRI:
    """load, read and process MRI images from .nib file"""

    def __init__(self, scan_file_path: Union[str, Path]) -> None:
        self.image_stack = self.__read_nib_file(scan_file_path)

    @staticmethod
    def __read_nib_file(file_path):
        img = nib.load(file_path)
        # Transform to canonical
        img_canonical = nib.as_closest_canonical(img)
        return img_canonical.get_fdata()

    def axis_view(self, idx):
        return self.image_stack[:, :, idx].T

    def coronal_view(self, idx):
        return self.image_stack[:, idx, :].T

    def sgattial_view(self, idx):
        return self.image_stack[idx, :, :].T

    def to_tensor(self):
        return torch.tensor(self.image_stack)