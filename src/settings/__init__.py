import os
import toml
import box

# Construct the absolute path to config.toml
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
config_path = os.path.join(ROOT_DIR, "config.toml")

# Load the settings
settings = box.Box(toml.load(config_path))
