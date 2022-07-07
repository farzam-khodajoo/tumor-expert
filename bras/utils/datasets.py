"""a collection of medical datasets"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from bras.utils.mri import MRI
from bras.utils.image import BraTsPreProcessing
from bras.nn.loss import expand_as_one_hot

class BraTs(Dataset, BraTsPreProcessing):

    MRI_MODALITIES = ["flair", "t1", "t1ce", "t2"]

    def __init__(self, path: Path, one_hot_encoding: bool=False) -> None:
        super().__init__()
        self.sample_list = self._list_samples(path=path)
        if len(self) == 0:
            print("warning, no image sample found in {}".format(str(path)))

        self.on_hot = one_hot_encoding

    @staticmethod
    def _list_samples(path):
        lsdir = list(Path(path).iterdir())
        print(f"loading {len(lsdir)} samples from {str(path)}")
        return lsdir

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
        data_batch = self.load_channels(index)
        channels, segmentation = self.crop_background(data_batch)
        if self.concate_one_hot_encoding: channels = self.concate_one_hot_encoding(channels)
        segmentation = expand_as_one_hot(segmentation)
        return channels, segmentation

    def __len__(self):
        return len(self.sample_list)
