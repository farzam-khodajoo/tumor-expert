import importlib
import numpy as np
import monai.transforms as transforms
import torch.nn.functional as F

def zero_padding(image_batch):
    # Determine maximum height and width
    # The mask's have the same height and width
    # since they mask the image.
    max_height = max([img.size(1) for img in image_batch])
    max_width = max([img.size(2) for img in image_batch])

    image_batch = [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in image_batch
    ]
    mask_batch = [
        # Same as for the images, but there is no channel dimension
        # Therefore the mask's width is dimension 1 instead of 2
        F.pad(mask, [0, max_width - mask.size(1), 0, max_height - mask.size(0)])
        for mask in mask_batch
    ]

    return image_batch, mask_batch


class BraTsPreProcessing:
    """crop redundant background voxels (with voxel value zero)"""

    @staticmethod
    def crop_background(images):
        """crop background pixel values, as they do not provide any useful information. to be ignored by the neural network"""
        input_images, segmentation = images
        bounding_box_image_sample = input_images[:, :, :, 100]
        crop_stack = np.concatenate([segmentation.reshape((1, *segmentation.shape)), input_images])
        bbox = transforms.utils.generate_spatial_bounding_box(bounding_box_image_sample)
        crop_image = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(crop_stack)
        segmentation = crop_image[0]
        input_images = crop_image[1:]
        return input_images, segmentation

    @staticmethod
    def concate_one_hot_encoding(input_channels: np.array):
        mask = np.ones(input_channels.shape[1:], dtype=np.float32)
        for idx in range(input_channels.shape[0]): mask[np.where(input_channels[idx] <= 0)] *= 0.0
        mask = np.expand_dims(mask, 0)
        return np.concatenate([input_channels, mask])

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