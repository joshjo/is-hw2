import numpy as np


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(
        features, target, num_steps, learning_rate, add_intercept=True):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    # lamb = 0.1

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        # reg = lamb / target.size * weights

        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

    return weights


def predict(X, weights):
    data_with_intercept = np.hstack((
        np.ones((X.shape[0], 1)), X
    ))
    final_scores = np.dot(data_with_intercept, weights)
    pred = np.round(sigmoid(final_scores))
    return pred


if __name__ == '__main__':
    X = np.array([[1, 3, 4], [2, 6, 7], [6, 1, 2], [7, 2, 3]])
    Y = np.array([1, 1, 0, 0])

    weights = logistic_regression(X, Y, 100, 5e-5)

    print(predict(np.array([[1, 2, 5]]), weights))
    # print('->', weights)
