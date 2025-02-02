import os
import dill as pickle
from typing import Optional

from src.settings import ROOT_DIR
from src.utils import load_pickle


class CheckpointManager(object):
    """
    A manager for saving, loading, listing, and deleting serialized checkpoints.

    Attributes:
        checkpoint_folder (str): The directory where checkpoints are stored.
    """

    CheckpointManagerError = type("CheckpointManagerError", (Exception,), {})

    def __init__(self, checkpoint_folder: str, sub_folder: str = None) -> None:
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_folder (str): Path to the folder where checkpoints will be stored.
        """
        assert checkpoint_folder, checkpoint_folder

        checkpoint_folder = (
            os.path.join(checkpoint_folder, sub_folder)
            if sub_folder
            else checkpoint_folder
        )

        self.checkpoint_folder = checkpoint_folder
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def _get_checkpoint_path(self, checkpoint_name: str) -> str:
        """
        Get the full path to a checkpoint file.

        Args:
            checkpoint_name (str): Name of the checkpoint file.

        Returns:
            str: Full path to the checkpoint file.
        """

        if not checkpoint_name.endswith(".pkl"):
            checkpoint_name += ".pkl"
        return os.path.join(self.checkpoint_folder, checkpoint_name)

    def save(self, state: dict, checkpoint_name: str) -> None:
        """
        Save a state dictionary to a checkpoint file.

        Args:
            state (dict): The state dictionary to save.
            checkpoint_name (str): The name of the checkpoint file.

        Raises:
            ValueError: If the state is not a dictionary.
            CheckpointManagerError: If saving the checkpoint fails.
        """

        checkpoint_path = self._get_checkpoint_path(checkpoint_name)
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as exc:
            raise self.CheckpointManagerError(
                f"Failed to save checkpoint '{checkpoint_name}'"
            ) from exc

    def load(self, checkpoint_path: str) -> Optional[dict]:
        """
        Load a state dictionary from a checkpoint file.

        Args:
            checkpoint_path (str): The path of the checkpoint file.

        Returns:
            dict: The loaded state dictionary.

        Raises:
            CheckpointManagerError: If loading the checkpoint fails.
        """
        # Checkpoint name cannot be nil
        assert checkpoint_path, checkpoint_path

        # Use EAFP style rather than checking if the file exists and so on...
        try:
            return load_pickle(checkpoint_path)
        except Exception as exc:
            raise self.CheckpointManagerError(
                f"Failed to revive checkpoint '{checkpoint_path}'"
            ) from exc

    def list(self) -> list[str]:
        """
        List all available checkpoint files, including those in subdirectories.

        Returns:
            list: A list of checkpoint file paths.
        """
        checkpoint_files = []

        try:
            absolute_dir = os.path.join(ROOT_DIR, self.checkpoint_folder)

            # Walk through all directories and subdirectories
            for root, _, files in os.walk(absolute_dir):
                for file in files:
                    if file.endswith(".pkl"):  # Check for .pkl files
                        # Get the full file path
                        full_path = os.path.join(root, file)
                        checkpoint_files.append(full_path)

            return checkpoint_files
        except Exception as exc:
            raise self.CheckpointManagerError(
                f"Failed to list checkpoints: {exc}"
            ) from exc

    def delete(self, checkpoint_name: Optional[str] = None) -> None:
        """
        Delete a checkpoint file.

        Args:
            checkpoint_name (str, optional): The name of the checkpoint file to delete.
                Defaults to the last saved checkpoint if not provided.

        Raises:
            CheckpointManagerError: If deleting the checkpoint fails or no checkpoints exist.
        """
        if not checkpoint_name:
            checkpoints = self.list()
            if not checkpoints:
                raise self.CheckpointManagerError("No checkpoints available to delete.")

            # Assuming the last checkpoint is the one with the most recent modification time.
            checkpoint_name = sorted(
                checkpoints,
            )[-1]

        checkpoint_path = self._get_checkpoint_path(checkpoint_name)

        try:
            os.remove(checkpoint_path)
        except Exception as exc:
            raise self.CheckpointManagerError(
                f"Failed to delete checkpoint `{checkpoint_name}`"
            ) from exc

    @property
    def last_created_name(self):
        """
        Get the last saved checkpoint file name.
        """
        checkpoints = self.list()
        if checkpoints:
            # This works for now
            return sorted(checkpoints)[-1]
