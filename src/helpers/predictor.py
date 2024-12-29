class Predictor(object):

    def __init__(self, func):
        self._func = func

    def predict(self, data):
        return self._func(data)
