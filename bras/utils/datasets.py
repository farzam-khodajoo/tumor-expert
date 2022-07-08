"""a collection of medical datasets"""
from modulefinder import Module
from pathlib import Path
from turtle import forward
from unicodedata import name
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms
from bras.utils.mri import MRI
from bras.nn.loss import expand_as_one_hot
import numpy as np


class BraTs(Dataset):

    MRI_MODALITIES = ["flair", "t1", "t1ce", "t2"]

    def __init__(self, path: Path, one_hot_encoding: bool = False, normalize: bool = True) -> None:
        super().__init__()
        self.sample_list = self._list_samples(path=path)
        if len(self) == 0:
            print("warning, no image sample found in {}".format(str(path)))

        self.on_hot = one_hot_encoding
        self.norm = transforms.NormalizeIntensity(
            nonzero=True, channel_wise=True)
        self.normalize_images = normalize

    @staticmethod
    def _list_samples(path):
        lsdir = list(Path(path).iterdir())
        print(f"loading {len(lsdir)} samples from {str(path)}")
        return lsdir

    @staticmethod
    def concate_one_hot_encoding(input_channels: np.array):
        mask = np.ones(input_channels.shape[1:], dtype=np.float32)
        for idx in range(input_channels.shape[0]):
            mask[np.where(input_channels[idx] <= 0)] *= 0.0
        mask = np.expand_dims(mask, 0)
        return np.concatenate([input_channels, mask])

    def load_channels(self, idx):
        sample_dir = self.sample_list[idx]
        sample_id = sample_dir.name

        channel_files = [
            sample_dir / f"{sample_id}_{file_tag}.nii.gz" for file_tag in self.MRI_MODALITIES]

        nifty_channels = torch.stack(
            [MRI(channel).to_tensor() for channel in channel_files])

        segmentation = MRI(sample_dir / f"{sample_id}_seg.nii.gz").to_tensor()

        return nifty_channels, segmentation

    def __getitem__(self, index: int):
        # load flair, t1, t2 and segmentation from each sample directory
        channels, segmentation = self.load_channels(index)
        if self.normalize_images:
            channels = self.norm(channels)
        if self.on_hot:
            channels = self.concate_one_hot_encoding(channels)
        # three channel output: edema, non-enhancing glioma, enhancing glioma
        channels = torch.tensor(channels)
        segmentation = torch.tensor(segmentation)
        return channels, segmentation

    def __len__(self):
        return len(self.sample_list)
