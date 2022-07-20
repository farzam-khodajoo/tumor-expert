import logging
from typing import List
from fastapi import APIRouter, UploadFile
from bras.app.components.nifty import NiftyViewManager

# dedicated to nifty file views
router = APIRouter()
manager = NiftyViewManager()


@router.post("/", tags=["views"])
async def upload_new_sample(t1_weight: UploadFile):
    try:
        logging.info("processing T1 weights")
        t1 = await manager.get_view(t1_weight)

        return {
            "t1": t1
        }

    except Exception as error:
        logging.warn(error)
        return {
            "message": error
        }
