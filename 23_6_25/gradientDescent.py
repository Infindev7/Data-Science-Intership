import numpy as np

class LinearRegression:
    def __init__(self, input_Data, input_label, b0=0, b1=0):
        self.b0, self.b1 = b0, b1
        self.X = input_Data
        self.y = input_label

    def fit(self):
        X_mean = np.mean(self.X)
        y_mean = np.mean(self.y)
        ssxy, ssx = 0,0
        for num in range(len(self.X)):
            ssxy += (self.X[num] - X_mean)*(self.y[num] - y_mean)
            ssx += (self.X[num] - X_mean) ** 2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - (self.b1 * X_mean)
        return self.b0, self.b1

    def predict(self, Xi):
        self.y_hat = self.b0 + (self.b1 * Xi)
        self.y_hat =  np.squeeze(self.y_hat)
        return self.y_hat

    def mean_squared_error(self):
        error = self.y - self.y_hat
        squared_error = error ** 2
        return np.mean(squared_error)

    def gredientDescent(self, alpha = 0.00005, epochs = 1):
        error = self.y - self.y_hat
        n = len(self.X)
        for i in range(epochs):
            del_b1 = (-2/n) * np.sum(self.X * error)
            del_b0 = (-2/n) * np.sum(error)
            self.b1 -= alpha * del_b1
            self.b0 -= alpha * del_b0
            print(f"Epoch No.: {i+1} | B1 : {self.b1} | B0 : {self.b0}")    #verbose
        return self.b0, self.b1


if __name__ == "__main__":
    heights = np.array([
        [160], [171], [182], [180], [154]
    ])

    weights = np.array([
        72, 76, 77, 83, 76
    ])

    LR = LinearRegression(input_Data=heights,input_label=weights)
    b0, b1 = LR.fit()
    print(f'The value of intercept:{b0}\nThe value of slope:{b1}')

    y_hat = LR.predict(heights)
    print(f'Value of Prediction of weight for height of cm: {y_hat}')

    loss = LR.mean_squared_error()
    print(f"Earlier Loss : {loss}")

    newb0, newb1 = LR.gredientDescent(epochs=10)
    print(newb0, newb1)

    new_y_hat = LR.predict(heights)
    print(f'New Predictions: {new_y_hat}')
    newmse = LR.mean_squared_error()
    print(f"new MSE: {newmse}")