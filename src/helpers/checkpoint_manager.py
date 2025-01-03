import os
import dill as pickle
from typing import Optional, Dict, List


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

    def save(self, state: Dict, checkpoint_name: str) -> None:
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

    def load(self, checkpoint_name: str) -> Optional[Dict]:
        """
        Load a state dictionary from a checkpoint file.

        Args:
            checkpoint_name (str): The name of the checkpoint file.

        Returns:
            dict: The loaded state dictionary.

        Raises:
            CheckpointManagerError: If loading the checkpoint fails.
        """
        # Checkpoint name cannot be nil
        assert checkpoint_name, checkpoint_name

        checkpoint_path = self._get_checkpoint_path(checkpoint_name)

        # Use EAFP style rather than checking if the file exists and so on...
        try:

            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            raise self.CheckpointManagerError(
                f"Failed to load checkpoint '{checkpoint_name}'"
            ) from exc

    def list(self) -> List[str]:
        """
        List all available checkpoint files.

        Returns:
            list: A list of checkpoint filenames in the folder.
        """
        try:
            return [
                file
                for file in os.listdir(self.checkpoint_folder)
                if file.endswith(".pkl")
                and os.path.isfile(os.path.join(self.checkpoint_folder, file))
            ]
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
            checkpoint_name = max(
                checkpoints,
                key=lambda x: os.path.getmtime(self._get_checkpoint_path(x)),
            )

        checkpoint_path = self._get_checkpoint_path(checkpoint_name)

        try:
            os.remove(checkpoint_path)
        except Exception as exc:
            raise self.CheckpointManagerError(
                f"Failed to delete checkpoint '{checkpoint_name}'"
            ) from exc
