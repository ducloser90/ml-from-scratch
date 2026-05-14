import numpy as np

class LogisticRegression():

    def __init__(self, lr=0.001, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.Weights = None
        self.Bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.Weights = np.zeros(n_features)
        self.Bias = 0
        for _ in range (self.max_iter):
            y_pred = np.dot(X, self.Weights) + self.Bias
            h = 1 / (1 + np.exp(-y_pred))
            dw = (1 / n_samples) * np.dot(X.T, (h - y))
            db = (1 / n_samples) * np.sum(h - y)
            self.Weights = self.Weights - dw * self.lr
            self.Bias = self.Bias - db * self.lr

    def predict(self, X):
        y_pred = np.dot(X, self.Weights) + self.Bias
        prob = 1 / (1 + np.exp(-y_pred))
        return np.array([1 if y>= 0.5 else 0 for y in prob])
    
    def accuracy_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)