import numpy as np
from regression import LogisticRegression
from process import stream_docs, get_minibatch, feature_extractor
from custom_regression import logistic_regression, predict


if __name__ == '__main__':
    size = 1000
    # model = LogisticRegression(alpha=0.1, num_iter=100)
    doc_stream = stream_docs(path='example/shuffled_movie_data.csv')
    reviews, sentiments = get_minibatch(doc_stream, size)
    X = []
    for i, review in enumerate(reviews):
        if not i % 1000:
            print("i", i)
        X.append(feature_extractor(review))

    weights = logistic_regression(
        np.array(X), np.array(sentiments), 100000, 5e-5)

    correct = 0
    incorrect = 0
    for x, y in zip(X, sentiments):
        pred = predict(np.array([x]), weights)
        if pred == y:
            correct += 1
        else:
            incorrect += 1

    print(correct, incorrect)
    # predict(np.ara)
    # model.fit(np.array(X), np.array(sentiments))


    # print('->', model.predict(np.array(
    #     [feature_extractor("I love this movie")]
    # )))
