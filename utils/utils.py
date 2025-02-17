import importlib
from pathlib import Path

import yaml
from addict import Dict


def read_yaml(file_path: str):
    with open(file_path, mode="r") as F:
        yml = yaml.load(F, Loader=yaml.Loader)
    return Dict(yml)


def check_path(file_path: str):
    if Path(file_path).is_file():
        return True
    else:
        raise ValueError(f"file {file_path} does not exist or is broken")


def dynamic_import(
    module_name: str = "models.trans_mil", attr_name: str = "TransMIL"
):
    try:
        module = importlib.import_module(module_name)
        module_class = getattr(module, attr_name)
        return module_class
    except ModuleNotFoundError:
        print(f"Module '{module_name}' not found. Installing...")
