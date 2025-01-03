class Predictor(object):

    def __init__(self, predict_func, render_func):
        self._predict = predict_func
        self._render = render_func

    def predict(self, data):
        return self._predict(data)

    def render(self, predictions):
        return self._render(predictions)
