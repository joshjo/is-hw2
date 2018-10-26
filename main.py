from regression import LogisticRegression
from process import *


if __name__ == '__main__':
    size = 1000
    model = LogisticRegression(alpha=0.1, num_iter=100)
    doc_stream = stream_docs(path='example/shuffled_movie_data.csv')
    reviews, sentiments = get_minibatch(doc_stream, size)
    X = []
    for i, review in enumerate(reviews):
        # if not i % 1000:
        #     print("i", i)
        X.append(feature_extractor(review))
    model.fit(np.array(X), np.array(sentiments))


    print('->', model.predict(np.array(
        [feature_extractor("I love this movie")]
    )))
