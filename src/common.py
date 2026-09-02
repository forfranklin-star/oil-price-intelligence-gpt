from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def root_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
