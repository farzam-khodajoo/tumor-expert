"""a collection of medical datasets"""
import torch
from torch.utils.data import Dataset
from bras.utils.image import MRI
from pathlib import Path
import numpy as np


class BraTsDataset(Dataset):
    """
    BraTS is a dataset which provides multimodal 3D brain MRIs and ground truth brain tumor segmentations annotated by physicians, consisting of 4 MRI modalities per case (T1, T1c, T2, and FLAIR).
    Annotations include 3 tumor subregions—the enhancing tumor, the peritumoral edema, and the necrotic and non-enhancing tumor core. The annotations were combined into 3 nested subregions—whole tumor (WT), tumor core (TC), and enhancing tumor (ET).
    The data were collected from 19 institutions, using various MRI scanners

    more on https://paperswithcode.com/dataset/brats-2018-1
    """

    NOT_TUMOR = 0
    NON_ENHANCING_TUMOR = 1
    EDEMA = 2
    ENHANCING_TUMOR = 4

    MRI_MODALITIES = ["flair", "t1", "t1ce", "t2"]

    def __init__(self, dataset_path: Path, expand_segmentations: bool = True, append_on_hot_channel: bool = True) -> None:
        super().__init__()
        # list of dataset samples into pytorch list
        self.sample_list = self._list_samples(path=dataset_path)

        self.expand_segmentations = expand_segmentations
        self.append_on_hot_channel = append_on_hot_channel

    @staticmethod
    def _list_samples(path):
        lsdir = list(Path(path).iterdir())
        print(f"loading {len(lsdir)} samples from {str(path)}")
        return lsdir

    def load_channels(self, idx):
        """Read samples from dataset and return mri channels (T1, T2, FLAIR, T1CE) and corresponding segmentation.
        """
        sample_dir = self.sample_list[idx]
        sample_id = sample_dir.name

        channel_files = [
            sample_dir / f"{sample_id}_{file_tag}.nii.gz" for file_tag in self.MRI_MODALITIES]

        nifty_channels = torch.stack(
            [MRI(channel).to_tensor() for channel in channel_files])

        segmentation = MRI(sample_dir / f"{sample_id}_seg.nii.gz").to_tensor()

        return nifty_channels, segmentation

    @staticmethod
    def concate_one_hot_encoding(input_channels: np.array):
        """
        To distinguish between background voxels and normalized voxels which have values close to zero,
        we add an input channel with one-hot encoding for foreground voxels and stacked with the input data.
        As a result, each example has 5 channels.

        output channels: T1, T2, FLAIR, T1CE, on_hot.
        """

        mask = np.ones(input_channels.shape[1:], dtype=np.float32)
        for idx in range(input_channels.shape[0]):
            mask[np.where(input_channels[idx] <= 0)] *= 0.0
        mask = np.expand_dims(mask, 0)
        return np.concatenate([input_channels, mask])

    def expand_on_hot_segmentation(self, seg: np.array):
        """segmentation consist of 4 pixel value: 0 for non-tumor, 1 for non-enhancing, 2 for edema and 4 for enhancing tumor.
        segmentation data of shape (240, 240, 155) will be transformed into (3, 240, 240, 155) to seperate each channel.
        """

        width, height, samples = seg.shape
        sliced_segmentation = []
        segmentations = [self.NON_ENHANCING_TUMOR,
                         self.ENHANCING_TUMOR, self.EDEMA]

        for segment_index in segmentations:
            mask = np.zeros((width, height, samples))
            idxs = np.where(seg == segment_index)
            mask[idxs] = 1
            sliced_segmentation.append(mask)

        return np.array(sliced_segmentation)

    def __getitem__(self, index):
        """ return 
        """
        channels, segmentations = self.load_channels(index)

        if self.append_on_hot_channel:
            channels = self.concate_one_hot_encoding(input_channels=channels)

        if self.expand_segmentations:
            segmentations = self.expand_on_hot_segmentation(segmentations)

        T = torch.tensor
        return T(channels), T(segmentations)

    def __len__(self):
        return len(self.sample_list)
