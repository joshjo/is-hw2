import numpy as np
import math
from nltk.stem.porter import PorterStemmer
import re
from nltk import bigrams

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

sia = SentimentIntensityAnalyzer()


stop = stopwords.words('english')
porter = PorterStemmer()
pronouns = 'i me we us my mine our ours you your yours'.split(' ')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return tokenized


def count_pronouns(words_list):
    occurs = [1 for word in words_list for pronoun in pronouns if word == pronoun]
    return len(occurs)


def exists_word(word, words_list):
    return 1 if word in words_list else 0


def feature_extractor(text):
    tokenized = tokenizer(text)
    features = defaultdict(int)
    tokenized = bigrams(tokenized)
    for word in tokenized:
        polarity = sia.polarity_scores(word[0] + ' ' + word[1])
        features['neg'] += polarity['neg']
        features['pos'] += polarity['pos']
    words_list = text.lower().split(' ')
    features['nointext'] = exists_word('no', words_list)
    features['exclamation'] = exists_word('!', words_list)
    features['pronouns'] = count_pronouns(words_list)
    features['wordcount'] = math.log(len(words_list))
    return np.array(list(features.values()))


def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return np.array(docs), np.array(y)


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


# if __name__ == '__main__':
#     analyse(5000)
