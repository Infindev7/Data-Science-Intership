import numpy as np
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self):
        self.b0, self.b1 = 0, 0

    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        ssxy, ssx = 0,0
        for num in range(len(X)):
            ssxy += (X[num] - X_mean)*(y[num] - y_mean)
            ssx += (X[num] - X_mean) ** 2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - (self.b1 * X_mean)
        return self.b0, self.b1

    def predict(self, Xi):
        y_hat = self.b0 + (self.b1 * Xi)
        return np.squeeze(y_hat)


if __name__ == "__main__":
    heights = np.array([
        [160], [171], [182], [180], [154]
    ])

    weights = np.array([
        72, 76, 77, 83, 76
    ])

    LR = LinearRegression()
    b0, b1 = LR.fit(X=heights,y=weights)
    #print(f'The value of intercept:{b0}\nThe value of slope:{b1}')

    y_hat = LR.predict(Xi=heights)
    #print(f'Value of Prediction of weight for height of cm: {y_hat}')

    print(f'True Label: {weights}')
    print(f'Predicted Label: {y_hat}')

    def meanSquaredError(y_true, y_pred):
        error = y_true - y_pred
        squaredError = error**2
        return np.mean(squaredError)

    mse = meanSquaredError(y_true=weights,y_pred=y_hat)
    print(mse)

    skmse = mean_squared_error(y_true=weights,y_pred=y_hat)
    print(skmse)