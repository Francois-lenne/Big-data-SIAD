from sklearn.linear_model import LogisticRegression
import numpy as np

class ClassificationModel:
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs')
        self.trained = False
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 1, 0, 1])

    def train(self):
        self.model.fit(self.X, self.y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise Exception("Model not trained yet.")
        y_pred = self.model.predict(X)
        return y_pred


