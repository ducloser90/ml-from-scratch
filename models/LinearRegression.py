import numpy as np

class LinearRegression():
    
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.Weights = None
        self.Bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.Weights = np.zeros(n_features)
        self.Bias = 0
        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.Weights) + self.Bias
            dw = (1/n_samples) * 2 * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * 2 * np.sum((y_pred - y))
            self.Weights = self.Weights - dw * self.lr
            self.Bias = self.Bias - db * self.lr

    def predict(self, X):
        return np.dot(X, self.Weights) + self.Bias
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)