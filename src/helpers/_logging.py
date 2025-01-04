import logging
import sys

from src.settings import settings

# Create a logger
logger = logging.getLogger()

# Set the logging level
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler for logging to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for logging to a file
file_handler = logging.FileHandler(settings.general.LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
