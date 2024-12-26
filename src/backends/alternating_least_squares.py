from ..utils.checkpoint_manager import CheckpointManager
from .core import BaseBackend


class AlternatingLeastSquares(BaseBackend):
    checkpoint_manager = CheckpointManager(
        checkpoint_folder="ALS_CHECKPOINT_FOLDER"
    )  # TODO
    pass
