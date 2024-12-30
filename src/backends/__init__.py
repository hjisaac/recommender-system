import logging
from src.utils import convert_flat_dict_to_string

logger = logging.getLogger(__name__)


class Backend(object):
    """
    This class is the director of all the other classes. It mainly interconnects
    all the components together and ensures their communication with each other.
    """

    def __init__(self, algorithm, checkpoint_manager, resume_enabled=False):
        self.algorithm = algorithm
        self.checkpoint_manager = checkpoint_manager
        self.resume_enabled = resume_enabled

    def __call__(self, data):

        if self.resume_enabled:
            self.algorithm.set_state(self.checkpoint_manager.load())

        self.algorithm.run(data=data, resume=self.resume_enabled)
        # Save the current state of the training from the algorithm object
        self.checkpoint_manager.save(
            self.algorithm.state, checkpoint_name=self._get_checkpoint_name()
        )
        logger.info(f"Checkpoint successfully saved at {self._instance_id}")

        return self.algorithm.state.to_predictor()


def _get_checkpoint_name(self):
    return convert_flat_dict_to_string(
        self.algorithm.state.intrinsic,
    )
