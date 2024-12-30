import logging
from src.utils import convert_flat_dict_to_string

logger = logging.getLogger(__name__)


class Backend(object):
    """
    This class is the director of all the other classes. It mainly interconnects
    all the components together and ensures their communication with each other.
    """

    def __init__(self, algorithm, checkpoint_manager):
        self.algorithm = algorithm
        self.checkpoint_manager = checkpoint_manager

    def __call__(self, data):
        self.algorithm.run(data=data)
        # Save the current state of the training from the algorithm object
        checkpoint_name = self._get_checkpoint_name()
        self.checkpoint_manager.save(self.algorithm.state, checkpoint_name)
        # TODO: Clean this
        # self.checkpoint_manager.save(
        #     {"loss_test": loss_test, "loss_train": loss_train}, self.__model_name
        # )
        logger.info(f"Checkpoint successfully saved at {self._instance_id}")

        return self.algorithm.state.to_predictor()


def _get_checkpoint_name(self):
    return convert_flat_dict_to_string(
        self.algorithm.state.intrinsic,
    )
