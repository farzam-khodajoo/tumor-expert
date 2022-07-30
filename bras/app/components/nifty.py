"""Utilities dedicated to files (.nii)"""
import logging
import uuid
import shutil
import aiofiles
import numpy as np
from pathlib import Path
from fastapi import UploadFile
from bras.nn.vino import VinoBraTs, normalize
from bras.utils.image import MRI
from monai.inferers import sliding_window_inference
import nibabel as nib
import torch


class NiftyFileManager:

    temporary_directory: str

    async def save_and_return_path(self, file: UploadFile):
        """
        save files into temporary directory to get later get loaded
        this approach is temporary for later be replaced with file-like object reader.
        """
        random_directory_name = uuid.uuid4()

        Path(self.temporary_directory).mkdir(exist_ok=True)
        dir_path = Path(self.temporary_directory) / str(random_directory_name)

        # create directory if not exists
        dir_path.mkdir(exist_ok=True)
        file_path = dir_path/file.filename

        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)

        logging.info("uploaded file has written to temporary file")
        return file_path

    def remove_temporary_directory(self):
        if Path(self.temporary_directory).exists():        
            logging.info("clear temporary directory")
            shutil.rmtree(self.temporary_directory)


class NiftyFileReader:

    def read_nib(self, path):
        logging.info("read nifty data from {}".format(path))
        return nib.load(path).get_fdata()

    def return_stack_and_normalize(self, t1, t1ce, t2, flair):
        def T(x): return normalize(
            image=x,
            mask=x > 0,
            full_intensities_range=True
        )

        return np.stack([
            T(t1), T(t1ce), T(t2), T(flair)
        ])

    def merge_segmentations(self, segmentations: np.array):
        mask = np.zeros(segmentations.shape[1:])
        for idx, channel in enumerate(segmentations, start=1):
            mask[np.where(channel > 0.62)] = idx

        return mask

class SegmentationController(NiftyFileManager, NiftyFileReader, VinoBraTs):
    def __init__(self, path_to_onnx, temporary_directory) -> None:
        super().__init__(
            path_to_onnx=path_to_onnx
        )
        self.temporary_directory = temporary_directory

    @staticmethod
    def sigmoid_function(z):
        """ this function implements the sigmoid function, and 
        expects a numpy array as argument """
        sigmoid = 1.0/(1.0 + np.exp(-z))
        return sigmoid 

    async def process_segmentation(self, t1, t1ce, t2, flair):
        #self.remove_temporary_directory()

        async def process_nib(x):
            path = await self.save_and_return_path(x)
            return self.read_nib(path=path)

        t1_weights = await process_nib(t1)
        t1ce_weights = await process_nib(t1ce)
        t2_weights = await process_nib(t2)
        flair_weights = await process_nib(flair)

        logging.info("stacking all input weights..")
        model_inputs = self.return_stack_and_normalize(
            t1=t1_weights,
            t1ce=t1ce_weights,
            t2=t2_weights,
            flair=flair_weights
        )


        logging.info("feed inputs into segmentation model")
        T = lambda x: torch.Tensor(x)
        model_output = sliding_window_inference(
            inputs=T(model_inputs.reshape((1, *model_inputs.shape))),
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor= lambda x: T(self.inference(x.numpy())),
            overlap=0.5
        )

        model_output = torch.sigmoid(model_output)
        segmentation = self.merge_segmentations(model_output[0])
        ni_img = nib.Nifti1Image(segmentation, affine=np.eye(4))
        
        save_path = Path(self.temporary_directory) / "result.nii.gz"
        nib.save(ni_img, save_path)
        return save_path
        


