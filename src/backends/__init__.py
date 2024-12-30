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
            # TODO:
            # self.algorithm.set_state(self.checkpoint_manager.load())
            pass

        self.algorithm.run(data=data, resume=self.resume_enabled)
        checkpoint_name = self._get_checkpoint_name()
        # Save the current state of the training from the algorithm object
        self.checkpoint_manager.save(
            self.algorithm.state.to_dict(), checkpoint_name=self._get_checkpoint_name()
        )
        logger.info(f"Checkpoint successfully saved at {checkpoint_name}")
        return self.algorithm.state.to_predictor()

    def _get_checkpoint_name(self):
        """Side effect method that returns a different string made
        of the hyperparameters for each call
        """
        return convert_flat_dict_to_string(
            {
                k.removeprefix("hyper_"): v
                for k, v in self.algorithm.state.to_dict().items()
                if k in self.algorithm.hyper_parameters
            }
        )
