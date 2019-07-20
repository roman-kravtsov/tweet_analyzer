import nltk
import numpy as np
from gensim.models import Word2Vec


def _get_word2vec_features(x, word2vec, all_words_per_tweet, max_tweet_len):
    """Computes features for a single tweet"""

    features = np.zeros((len(x), max_tweet_len, word2vec.vector_size))

    for i, tweet_words in enumerate(all_words_per_tweet):
        tweet_repr = np.array(
            [word2vec.wv[r] if r in word2vec.wv.vocab else np.zeros(word2vec.vector_size) for r in tweet_words])
        features[i][:len(tweet_repr), :word2vec.vector_size] = tweet_repr

    return features


def get_word2vec_features(x_train, x_test):
    """Prepares Word2Vec features for train and test data"""

    all_words_per_tweet_train = [nltk.word_tokenize(sent) for sent in x_train["text"]]
    all_words_per_tweet_test = [nltk.word_tokenize(sent) for sent in x_test["text"]]

    word2vec = Word2Vec(all_words_per_tweet_train, min_count=5)
    word2vec.train(all_words_per_tweet_train, total_examples=word2vec.corpus_count, epochs=15)

    max_tweet_len = np.max(
        [np.max([len(t) for t in all_words_per_tweet_train]), np.max([len(t) for t in all_words_per_tweet_test])])

    features_train = _get_word2vec_features(x_train, word2vec, all_words_per_tweet_train, max_tweet_len)
    features_test = _get_word2vec_features(x_test, word2vec, all_words_per_tweet_test, max_tweet_len)

    return features_train, features_test


def update_labels(y_train, y_test):
    """Updates labels in case they are not 0s and 1s"""

    train_labels_updated = []
    test_labels_updated = []

    for lbl in y_train["polarity"]:
        train_labels_updated.append(0 if lbl == 0 else 1)

    for lbl in y_test["polarity"]:
        test_labels_updated.append(0 if lbl == 0 else 1)

    return train_labels_updated, test_labels_updated
