import unittest
import os
from unittest.mock import patch, MagicMock
import dill as pickle
from src.helpers.checkpoint_manager import CheckpointManager


class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        """ Set up test environment """
        self.test_folder = "test_checkpoints"
        self.sub_folder = "subdir"
        self.manager = CheckpointManager(self.test_folder, self.sub_folder)
        self.test_file_name = "test_checkpoint"
        self.state = {"key1": "value1", "key2": "value2"}
        self.test_checkpoint_path = os.path.join(
            self.test_folder, self.sub_folder, f"{self.test_file_name}.pkl"
        )

        # Ensure the directory exists for testing
        os.makedirs(os.path.join(self.test_folder, self.sub_folder), exist_ok=True)

    def tearDown(self):
        """ Clean up after tests """
        if os.path.exists(self.test_checkpoint_path):
            os.remove(self.test_checkpoint_path)

        # Clean up the test folder
        if os.path.exists(self.test_folder):
            for root, dirs, files in os.walk(self.test_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

    @patch("builtins.open", MagicMock())
    def test_save_checkpoint_success(self):
        """ Test saving a checkpoint """
        self.manager.save(self.state, self.test_file_name)
        self.assertTrue(os.path.exists(self.test_checkpoint_path))
        with open(self.test_checkpoint_path, "rb") as f:
            loaded_state = pickle.load(f)
        self.assertEqual(loaded_state, self.state)

    def test_save_checkpoint_invalid_state(self):
        """ Test saving a checkpoint with an invalid state """
        with self.assertRaises(ValueError):
            self.manager.save([], self.test_file_name)

    @patch("builtins.open", MagicMock())
    def test_load_checkpoint_success(self):
        """ Test loading a checkpoint """
        # First save the checkpoint
        self.manager.save(self.state, self.test_file_name)

        # Now load it
        loaded_state = self.manager.load(self.test_file_name)
        self.assertEqual(loaded_state, self.state)

    def test_load_checkpoint_not_found(self):
        """ Test loading a non-existent checkpoint """
        with self.assertRaises(FileNotFoundError):
            self.manager.load("non_existent_checkpoint.pkl")

    def test_load_checkpoint_invalid_format(self):
        """ Test loading a corrupted checkpoint """
        # Corrupt the file (create an empty file)
        with open(self.test_checkpoint_path, "wb") as f:
            f.write(b"")

        with self.assertRaises(Exception):
            self.manager.load(self.test_file_name)

    @patch("os.listdir", return_value=["test_checkpoint.pkl"])
    def test_list_checkpoints(self):
        """ Test listing available checkpoints """
        checkpoints = self.manager.list()
        self.assertIn("test_checkpoint.pkl", checkpoints)

    @patch("os.listdir", return_value=[])
    def test_list_checkpoints_empty(self):
        """ Test listing when no checkpoints are available """
        checkpoints = self.manager.list()
        self.assertEqual(checkpoints, [])

    def test_delete_checkpoint_success(self):
        """ Test deleting a checkpoint """
        # Save the checkpoint
        self.manager.save(self.state, self.test_file_name)

        # Delete the checkpoint
        self.manager.delete(self.test_file_name)
        self.assertFalse(os.path.exists(self.test_checkpoint_path))

    def test_delete_checkpoint_not_found(self):
        """ Test deleting a non-existent checkpoint """
        with self.assertRaises(FileNotFoundError):
            self.manager.delete("non_existent_checkpoint.pkl")

    @patch("os.remove", MagicMock())
    def test_delete_checkpoint_fail(self):
        """ Test failure to delete a checkpoint due to system error """
        # Mock os.remove to raise an exception
        os.remove.side_effect = Exception("Deletion failed")

        # Save the checkpoint first
        self.manager.save(self.state, self.test_file_name)

        with self.assertRaises(Exception):
            self.manager.delete(self.test_file_name)

    @patch("os.makedirs", MagicMock())
    def test_create_directory_on_init(self):
        """ Test that the checkpoint folder is created if it does not exist """
        # Mock os.makedirs to avoid actual folder creation
        manager = CheckpointManager(self.test_folder)
        self.assertTrue(os.path.exists(self.test_folder))

    def test_assert_checkpoint_folder(self):
        """ Test if the folder assertion works when the folder is invalid """
        with self.assertRaises(AssertionError):
            CheckpointManager("", self.sub_folder)

    def test_get_checkpoint_path(self):
        """ Test _get_checkpoint_path functionality """
        checkpoint_path = self.manager._get_checkpoint_path(self.test_file_name)
        expected_path = os.path.join(self.test_folder, self.sub_folder, f"{self.test_file_name}.pkl")
        self.assertEqual(checkpoint_path, expected_path)

