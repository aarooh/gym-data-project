import io
from pathlib import Path
from typing import Union, Dict
import yaml

from pydantic import BaseModel


class Config(BaseModel):
    """Config handler class"""

    model_path: str
    gym_data_name: str
    weather_data_name: str
    device_mapping: Dict[str, str]


def get_config(path: Union[str, Path]) -> Config:
    """Load config from path"""
    cfg_path = Path(path).absolute()
    buffer = io.StringIO(cfg_path.read_text(encoding="utf-8"), newline="")
    config_data = yaml.safe_load(buffer)
    return Config(**config_data)
