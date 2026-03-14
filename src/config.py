import os
import yaml
from pathlib import Path

# Determine the directory of this file
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

# Load config once
config = load_config()

def get_config():
    """Get the configuration dictionary."""
    return config
