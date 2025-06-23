import numpy as np

class LogisticRegression:
    def __init__(self):
        self.b0, self.b1 = 0, 0
        self.sigmoid = np.array([])

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

    #def predict(self, Xi):
    #    z = self.b0 + (self.b1 * Xi)
    #    sigmoid = 1/ (1 + np.exp(-z))
    #    if sigmoid >= 0.5:
    #        y_pred = 1
    #    else:
    #        y_pred = 0
    #    return sigmoid, y_pred
    
    def predict(self, Xi):
        z = self.b0 + (self.b1 * Xi)
        sigmoidFunction = [1/(1 + np.exp(-z))]
        self.sigmoid = np.append(self.sigmoid, sigmoidFunction)
        return self.sigmoid

if __name__ == "__main__":
    X = np.array([
        [0.5], [1.5], [2], [4.25], [3.25], [5.50]
    ])

    y = np.array([
        0, 0, 0, 1, 1, 1
    ])

    LR = LogisticRegression()
    b0, b1 = LR.fit(X=X,y=y)
    #print(f'The value of intercept:{b0}\nThe value of slope:{b1}')

    #print(f'Value of Prediction of weight for height of cm: {y_hat}')

    #print(f'True Label: {weights}')
    #def meanSquaredError(y_true, y_pred):
    #    error = y_true - y_pred
    #    squaredError = error**2
    #    return np.mean(squaredError)

    sigmoid= LR.predict(X)
    print(sigmoid)

    result = list()
    for i in range(len(sigmoid)):
        if sigmoid[i] >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        result.append(y_pred)

    print(f"True Label: {y}")
    print(f"Pred Label: {result}")
    #mse = meanSquaredError(y_true=weights,y_pred=y_hat)
    #print(mse)
#