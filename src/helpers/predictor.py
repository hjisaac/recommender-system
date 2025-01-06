class Predictor(object):

    def __init__(self, predict_func, render_func):
        self.predict_func = predict_func
        self.render_func = render_func

    def predict(self, data):
        return self.predict_func(data)

    def render(self, predictions):
        return self.render_func(predictions)
