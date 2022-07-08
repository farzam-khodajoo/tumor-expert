import torch
from monai import transforms


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
