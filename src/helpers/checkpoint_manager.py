import os
import dill as pickle
from typing import Optional, Dict, List


class CheckpointManager(object):
    """
    A manager for saving, loading, listing, and deleting serialized checkpoints.

    Attributes:
        checkpoint_folder (str): The directory where checkpoints are stored.
    """

    def __init__(self, checkpoint_folder: str):
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_folder (str): Path to the folder where checkpoints will be stored.
        """

        # Checkpoint folder can not be nil
        assert checkpoint_folder, checkpoint_folder

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
            Exception: If saving the checkpoint fails.
        """
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary.")
        checkpoint_path = self._get_checkpoint_path(checkpoint_name)
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            raise Exception(f"Failed to save checkpoint '{checkpoint_name}': {e}")

    def load(self, checkpoint_name: str) -> Optional[Dict]:
        """
        Load a state dictionary from a checkpoint file.

        Args:
            checkpoint_name (str): The name of the checkpoint file.

        Returns:
            dict: The loaded state dictionary.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            Exception: If loading the checkpoint fails.
        """
        # Checkpoint name cannot be nil
        assert checkpoint_name, checkpoint_name

        checkpoint_path = self._get_checkpoint_path(checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found.")
        try:
            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load checkpoint '{checkpoint_name}': {e}")

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
        except Exception as e:
            raise Exception(f"Failed to list checkpoints: {e}")

    def delete(self, checkpoint_name: str) -> None:
        """
        Delete a checkpoint file.

        Args:
            checkpoint_name (str): The name of the checkpoint file to delete.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            Exception: If deleting the checkpoint fails.
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found.")
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            raise Exception(f"Failed to delete checkpoint '{checkpoint_name}': {e}")
