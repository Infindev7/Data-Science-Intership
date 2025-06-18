import numpy as np
from sklearn.linear_model import linearRegression

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
        return y_hat



if __name__ == "__main__":
    heights = np.array([
        [160], [171], [182], [180], [154]
    ])

    weights = np.array([
        72, 76, 77, 83, 76
    ])

    LR = LinearRegression()
    b0, b1 = LR.fit(X=heights,y=weights)
    print(f'The value of intercept:{b0}\nThe value of slope:{b1}')

    Xi = [[176]]
    y_hat = LR.predict(Xi)
    print(f'Value of Prediction of weight for height of {Xi}cm: {y_hat}')


    model = LinearRegression()
    model.fit(heights,weights)
    y_pred = model.predict(Xi)
    print(f'Value of Prediction of weight for height of {Xi}cm: {y_pred}')