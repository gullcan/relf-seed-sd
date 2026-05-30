from pathlib import Path
from typing import Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"


def load_paths(config_path: Union[str, Path, None] = None) -> dict:
    #

    if config_path is None:
        config_file = DEFAULT_CONFIG_PATH
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        paths = yaml.safe_load(file)

    return paths


def get_project_root(config_path: Union[str, Path, None] = None) -> Path:
    paths = load_paths(config_path)
    project_root = Path(paths["project_root"])

    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    return project_root


def get_dataset_root(config_path: Union[str, Path, None] = None) -> Path:
    paths = load_paths(config_path)
    dataset_root = Path(paths["dataset_root"])

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    return dataset_root