from os import environ
from pydantic import BaseSettings

class Settings(BaseSettings):
    SEGMENTATION_MODEL_WEIGHTS: str = environ.get("WEIGHTS")
    REACT_BUILD: str = environ.get("READ_BUILD")

settings = Settings()