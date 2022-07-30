from os import environ
from pydantic import BaseSettings

class Settings(BaseSettings):
    SEGMENTATION_MODEL_WEIGHTS: str = environ.get("WEIGHTS", "")
    REACT_BUILD: str = environ.get("READ_BUILD", "")
    TEMP_DIR: str = environ.get("TEMP", "temp_directory")

settings = Settings()