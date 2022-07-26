from functools import partial
from io import BytesIO
import logging
from shutil import ExecError
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
from bras.app.settings import settings
from bras.app.components.nifty import SegmentationController

# dedicated to nifty file views
router = APIRouter()
controller = SegmentationController(
    path_to_onnx=settings.SEGMENTATION_MODEL_WEIGHTS
)

@router.post("/", tags=["views"])
async def upload_new_sample(
    t1: UploadFile,
    t2: UploadFile,
    t1ce: UploadFile,
    flair: UploadFile
):
    logging.info("processing new segmentation request")
    try:
        nii_path = await controller.process_segmentation(
            t1=t1,
            t1ce=t1ce,
            t2=t2,
            flair=flair
        )

        return FileResponse(nii_path)

    except ExecError as error:
        logging.warning(error)
        return {
            "message": "error"
        }
