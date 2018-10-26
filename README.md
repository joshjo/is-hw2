
# Regresión Logística - Sentiment analysis

In the following project we implement a sentiment analysis using Logit Regression. The training data is obtained from IMDB movies.


## Features extractor


We use the following characteristics:

- Sum of neg:
- Sum of positive
- Sum of no words
- Sum of reviews with exclamation
- Sum of 1st, 2nd pronouns
- Log of sum of number of words


```python
from process import stream_docs, get_minibatch, feature_extractor
```


```python
size = 20000
```


```python
doc_stream = stream_docs(path='example/shuffled_movie_data.csv')
reviews, sentiments = get_minibatch(doc_stream, size)
```


```python
X = []
```


```python
for i, review in enumerate(reviews):
        if not i % 1000:
            print("i", i)
        X.append(feature_extractor(review))
```

    i 0
    i 1000
    i 2000
    i 3000
    i 4000
    i 5000
    i 6000
    i 7000
    i 8000
    i 9000
    i 10000
    i 11000
    i 12000
    i 13000
    i 14000
    i 15000
    i 16000
    i 17000
    i 18000
    i 19000



```python
from custom_regression import logistic_regression, predict
```


```python
import numpy as np
```

Getting the theta


```python
weights = logistic_regression(
        np.array(X), np.array(sentiments), 100000, 5e-5)
```


```python
correct = 0
incorrect = 0
for x, y in zip(X, sentiments):
    pred = predict(np.array([x]), weights)
    if pred == y:
        correct += 1
    else:
        incorrect += 1

print(correct, incorrect)
```

    11829 8171



```python
print("Accuracy of logit regression %.2f" % (correct / size))
```

    Accuracy of logit regression 0.59


## Regularization

we update the function with regularization


```python
def logistic_regression(
        features, target, num_steps, learning_rate, add_intercept=True):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    lamb = 0.01

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        reg = lamb / target.size * weights

        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient + reg

    return weights
```


```python
from custom_regression import logistic_regression, predict
```


```python
weights = logistic_regression(
        np.array(X), np.array(sentiments), 120000, 5e-4)
```


```python
correct = 0
incorrect = 0
for x, y in zip(X, sentiments):
    pred = predict(np.array([x]), weights)
    if pred >= 0.5:
        incorrect += 1
    else:
        correct += 1

print(correct, incorrect)
```

    17199 2801



```python
print("Accuracy of logit regression with regularization %.2f" % (correct / size))
```

    Accuracy of logit regression with regularization 0.86



```python

```
