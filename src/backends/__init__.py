import logging


logger = logging.getLogger(__name__)

something_else = ""
class Backend(object):
    """
    This class is the director of all the other classes. It mainly interconnects
    all the component together and ensure their communication with each other.
    """
    def __init__(self, algorithm, checkpoint_manager):
        self.algorithm = algorithm
        self.checkpoint_manager = checkpoint_manager

        self._instance_id = ""
        self._instance_name = algorithm.__class__.__name__ # ?

    def _get_model_name(self):
        return self._instance_name

    def __call__(self, data):
        self.algorithm.fit(data=data)
        # Save the current state of the training from the algorithm object
        # self.checkpoint_manager.save(self.algorithm.state, self._instance_name)
        # TODO: Clean this
        # self.checkpoint_manager.save(
        #     {"loss_test": loss_test, "loss_train": loss_train}, self.__model_name
        # )
        logger.log(f"Checkpoint successfully saved at {self._model_name}")

        self.checkpoint_manager.save(self.algorithm.state)

