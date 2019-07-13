import numpy as np

from keras.models import model_from_json
from gensim.models import Word2Vec

from server.static.preprocess import clean_tweet

root_path = "./server/models/"


def load_nn_model():
    # load json and create model
    with open(root_path + 'model.json', 'r') as model_file:
        loaded_model = model_from_json(model_file.read())

    # load weights into new model
    loaded_model.load_weights(root_path + "model.h5")

    return loaded_model


def load_word2vec_model():
    return Word2Vec.load(root_path + "word2vec.model")


def get_features(word2vec_model, tweet_words, max_tweet_len):
    w2v_vector_size = word2vec_model.vector_size

    features = np.zeros((max_tweet_len, w2v_vector_size))

    tweet_repr = np.array([word2vec_model.wv[r]
                           if r in word2vec_model.wv.vocab
                           else np.zeros(w2v_vector_size)
                           for r in tweet_words])

    features[:len(tweet_repr), :w2v_vector_size] = tweet_repr

    return features


w2vec = load_word2vec_model()
pred_model = load_nn_model()


def get_prediction_for_tweet(tweet):
    tweet = clean_tweet(tweet)
    tweet_words = tweet.split(" ")

    tweet_features = np.array([get_features(w2vec, tweet_words, pred_model.get_input_shape_at(0)[1])])
    prediction = pred_model.predict_classes(tweet_features)

    return prediction.flatten()[0]
