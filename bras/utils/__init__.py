
import yaml
from pathlib import Path

def read_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())