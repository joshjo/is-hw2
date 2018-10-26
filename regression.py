import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


class LogisticRegression:

    def __init__(self, alpha=0.01, num_iter=100000, fit_intercept=True):
        self.alpha = alpha
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        lamb = 0.01
        if self.fit_intercept:
            X = self.__add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            # reg = lamb / y.size * self.theta
            reg = 0
            gradient = np.dot(X.T, (h - y)) / y.size + reg
            self.theta -= self.alpha * gradient

            z = np.dot(X, self.theta)
            h = sigmoid(z)
            l = loss(h, y)

            # if(i % (self.num_iter / 10) == 0):
            #     print(f'loss: {l} \t')
                # print('reg', reg)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


if __name__ == '__main__':
    X = np.array([[1, 3, 4], [2, 6, 7], [6, 1, 2], [7, 2, 3]])
    Y = np.array([1, 1, 0, 0])
    lr = LogisticRegression()
    # for x, y in zip(X, Y):
    #     lr.fit(np.array([x]), np.array([y]))
    lr.fit(X, Y)
    print('-- -- -- --')
    print(lr.predict(np.array([[1, 2, 5]])))
