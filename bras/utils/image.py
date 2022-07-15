from typing import Union
from pathlib import Path
import nibabel as nib
import torch
import numpy as np
from monai import transforms


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


class CropBackground(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.crop = transforms.CropForeground(
            select_fn=lambda x: x > 0, margin=0, channel_indices=3)

    def forward(self, batch, *_):
        images, segmentations = batch
        bbox_start, bbox_end = self.crop.compute_bounding_box(images)
        print("s: {}, e: {}". format(bbox_start, bbox_end))
        cropped_images = self.crop.crop_pad(images, bbox_start, bbox_end)
        cropped_segmentation = self.crop.crop_pad(
            segmentations, bbox_start, bbox_end)
        return cropped_images, cropped_segmentation


def expand_segmentation_as_one_hot(segmentation_batch, indexes):
    """expand multi channel tensor (Batch, W, H, D) into multi-channel binary form (Batch, Channels, W, H, D)"""

    stacked_segmentations = []

    for segment in segmentation_batch.cpu().numpy():
        width, height, slides = segment.shape
        segment_channels = []

        for class_index in indexes:
            mask = torch.zeros((width, height, slides))
            idxs = np.where(segment == class_index)
            mask[idxs] = 1
            segment_channels.append(mask)

        segment_channels = torch.stack(segment_channels)
        stacked_segmentations.append(segment_channels)

    return torch.stack(stacked_segmentations)
