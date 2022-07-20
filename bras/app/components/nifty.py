"""Utilities dedicated to files (.nii)"""
from typing import List
from enum import Enum
import logging
from pathlib import Path
from fastapi import UploadFile
import nibabel as nib
import numpy as np
import cv2
import shutil
import aiofiles


class NiftyProcessor:
    """read .nii files and convert into stack of uint8 images"""

    def normalize_into_uint8(self, nib_3d: np.array):
        """normaize pixel values into range 0-255 and return as type np.uint8"""
        logging.info("normalizing nifty data to uint8")
        normalized_images: np.array = cv2.normalize(
            nib_3d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return normalized_images.astype(np.uint8)

    def reconstruct_stack(self, nib_3d: np.array) -> List:
        """convert (W, H, Stack) into (Stack, H, W) and return list type"""
        logging.info("re-construct nifty data")
        return np.moveaxis(nib_3d, -1, 0).tolist()


class NiftyFileManager:

    temp_directory = "./temporary_data_directory"

    async def save_and_return_path(self, file: UploadFile):
        """
        save files into temporary directory to get later get loaded
        this approach is temporary for later be replaced with file-like object reader.
        """

        dir_path = Path(self.temp_directory)

        # create directory if not exists
        dir_path.mkdir(exist_ok=True)
        file_path = dir_path/file.filename

        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)

        logging.info("uploaded file has written to temporary file")
        return file_path

    def remove_temporary_directory(self):
        logging.info("clear temporary directory")
        shutil.rmtree(self.temp_directory)


class NiftyFileReader:

    def read_nib(self, path):
        logging.info("read nifty data from {}".format(path))
        return nib.load(path).get_fdata()


class NiftyViewManager(
    NiftyProcessor,
    NiftyFileManager,
    NiftyFileReader
):
    def __init__(self) -> None:
        super().__init__()

    async def get_view(self, file: UploadFile):
        logging.info("start processing new nifty file upload..")
        # read file into numpy array
        nib_file_path = await self.save_and_return_path(file)
        nib_3d = self.read_nib(nib_file_path)
        # normalize to later be manipulated into image
        normalized_nib = self.normalize_into_uint8(nib_3d)
        view = self.reconstruct_stack(normalized_nib)

        self.remove_temporary_directory()
        return view
