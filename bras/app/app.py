import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from bras.app.settings import settings
from bras.app.routers import views
from pathlib import Path

logging.basicConfig(level=logging.INFO)

app = FastAPI()
logging.info("""
    react build path: {}
    onnx path: {}
""".format(
    settings.REACT_BUILD,
    settings.SEGMENTATION_MODEL_WEIGHTS
))

app.include_router(views.router, prefix="/views")
app.mount("/static", StaticFiles(directory=Path(settings.REACT_BUILD) / "static"), name="static")
app.mount("/", StaticFiles(directory=Path(settings.REACT_BUILD), html = True), name="static")