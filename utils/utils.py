import yaml


def read_yaml(file_path: str):
    with open(file_path, mode="r") as F:
        yml = yaml.load(F, Loader=yaml.Loader)
    return yml
