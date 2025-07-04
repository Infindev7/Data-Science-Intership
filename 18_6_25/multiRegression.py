import numpy as np
from sklearn.linear_model import LinearRegression

class MultiRegression:
    def __init__(self):
        self.params = np.zeros(
            int(np.random.random()), float
        )[:,np.newaxis]

    def fit(self, X, y):
        bias = np.ones(len(X))
        X_bias = np.c_[bias, X]
        #print(X_bias)
        inner_part = np.transpose(X_bias) @ X_bias
        #print(inner_part)
        inverse = np.linalg.inv(inner_part)
        #print(inverse)
        X_part = inverse @ np.transpose(X_bias)
        lse = X_part @ y
        self.params = lse
        return self.params

    def predict(self, Xi):
        bias_test = np.ones(len(Xi))
        X_test = np.c_[bias_test,Xi]
        y_hat = X_test @ self.params
        return y_hat

if __name__ == "__main__":
    X = np.array(
        [
            [1,4],
            [2,5],
            [3,8],
            [4,2]
        ]
    )

    y = np.array([1, 6, 8, 12])
    lr = MultiRegression()
    lr.fit(X,y)
    b_hat = lr.fit(X,y)
    print(b_hat)

    X_test = np.array(
        [
            [5,3]
        ]
    )

    y_hat = lr.predict(X_test)
    print('Hardcoded Model:',y_hat)


    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    print('SKlearn Model:',y_pred)