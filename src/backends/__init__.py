from src.utils import convert_flat_dict_to_string
from src.helpers._logging import logger  # noqa


class Backend(object):
    """
    This class is the director of all the other classes. It mainly interconnects
    all the components together and ensures their communication with each other.
    """

    def __init__(self, algorithm, checkpoint_manager, resume=False):
        self.algorithm = algorithm
        self.checkpoint_manager = checkpoint_manager
        self.resume = resume

    def __call__(self, data):
        initial_state = None

        if self.resume:
            initial_state = self.checkpoint_manager.load(
                self.checkpoint_manager.last_created_name,
            )
            logger.info(
                f"Checkpoint {self.checkpoint_manager.last_created_name} loaded with success",
            )

        # Run the algorithm
        self.algorithm.run(
            data=data,
            initial_state=initial_state,
        )

        # Get the checkpoint name under which to save the finalized state of the algorithm
        checkpoint_name = self._get_checkpoint_name()
        # Save the current state of the training from the algorithm object
        self.checkpoint_manager.save(
            self.algorithm.state, checkpoint_name=self._get_checkpoint_name()
        )

        logger.info(f"Checkpoint successfully saved at {checkpoint_name}")
        # Here, we're passing the algorithm to the `to_predictor` method, because to do
        # prediction we still need the algorithm. This is not a common case, but we're
        # designing the code to be generic and to be extremely extensible. It is clear
        # that we cannot access the algorithm from the state object. And It is not a good
        # idea to instantiate a new algorithm somewhere in the downstream code because of
        # some context that we will have to build again and complications it will come with.
        return self.algorithm.state.to_predictor(self.algorithm)

    def _get_checkpoint_name(self):
        """Side effect method that returns a different string made
        of the hyperparameters for each call
        """
        return convert_flat_dict_to_string(
            {
                k.removeprefix("hyper_"): getattr(self.algorithm.state, k)
                for k in self.algorithm.HYPER_PARAMETERS
            }
        )
