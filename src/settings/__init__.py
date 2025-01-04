import toml
import box
from .logging import configure_logging

settings = box.Box(toml.load("./config.toml"))
