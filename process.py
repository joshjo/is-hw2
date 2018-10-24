import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

sia = SentimentIntensityAnalyzer()


stop = stopwords.words('english')
porter = PorterStemmer()


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return tokenized


def feature_extractor(text):
    tokenized = tokenizer(text)
    total_polarity = defaultdict(int)
    tokenized = bigrams(tokenized)
    for word in tokenized:
        polarity = sia.polarity_scores(word[0] + ' ' + word[1])
        total_polarity['neg'] += polarity['neg']
        total_polarity['pos'] += polarity['pos']
        total_polarity['neu'] += polarity['neu']
    return total_polarity


def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def analyse(size):
    doc_stream = stream_docs(path='example/shuffled_movie_data.csv')
    reviews, sentiments = get_minibatch(doc_stream, size)
    correct = 0
    incorrect = 0
    for review, sentiment in zip(reviews, sentiments):
        polarity = feature_extractor(review)
        if sentiment and polarity['neg'] > polarity['pos']:
            incorrect += 1
        else:
            correct += 1
    print(correct, incorrect)


if __name__ == '__main__':
    analyse(5000)
