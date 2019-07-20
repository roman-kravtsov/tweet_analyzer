import numpy as np
import pickle

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

from static.preprocess import clean_tweet

cnn_path = "./models/w2v_cnn/"
cnn_model_file = "w2v_cnn_model.json"
cnn_model_weights_file = "w2v_cnn_weights.h5"
w2vec_model_file = "word2vec.model"

lstm_path = "./models/lstm/"
lstm_model_file = "lstm.json"
lstm_model_weights_file = "lstm.h5"
lstm_tokenizer_model_file = "tokenizer.model"

nb_path = "./models/tfidf_nb/"
nb_base_file_name = "nb_prep_data_50k_train_10k_test"

svc_path = "./models/tfidf_svc/"
svc_base_file_name = "svc_prep_data_10k_train_2k_test"


def load_word2vec_and_cnn_model():
    """Loads CNN model and weights, loads Word2Vec model"""

    word2vec_model = Word2Vec.load(cnn_path + w2vec_model_file)

    with open(cnn_path + cnn_model_file, 'r') as model_reader:
        loaded_model = model_from_json(model_reader.read())
    loaded_model.load_weights(cnn_path + cnn_model_weights_file)

    return word2vec_model, loaded_model


def load_vectorizer_and_lstm_model():
    """Loads CNN model and weights, loads Word2Vec model"""

    with open(lstm_path + lstm_tokenizer_model_file, 'rb') as reader:
        tokenizer = pickle.load(reader)

    with open(lstm_path + lstm_model_file, 'r') as model_reader:
        loaded_model = model_from_json(model_reader.read())
    loaded_model.load_weights(lstm_path + lstm_model_weights_file)

    return tokenizer, loaded_model


def load_tfidf_and_clf_model(path, name):
    """Loads TF-IDF and classifier model"""

    with open("{}{}{}".format(path, name, ".vect"), 'rb') as f:
        tfidf_vect = pickle.load(f)

    with open("{}{}{}".format(path, name, ".clf"), 'rb') as f:
        clf = pickle.load(f)

    return tfidf_vect, clf


def get_features_w2v(word2vec_model, tweet_words, max_tweet_len):
    """Creates features from tweet based on Word2Vec model"""

    w2v_vector_size = word2vec_model.vector_size

    # cut tweet if it is longer than the longest tweet
    # we encountered during training
    if len(tweet_words) > max_tweet_len:
        tweet_words = tweet_words[:max_tweet_len]

    features = np.zeros((max_tweet_len, w2v_vector_size))

    tweet_repr = np.array([word2vec_model.wv[r]
                           if r in word2vec_model.wv.vocab
                           else np.zeros(w2v_vector_size)
                           for r in tweet_words])

    features[:len(tweet_repr), :w2v_vector_size] = tweet_repr

    return features


def get_features_tokenizer(tokenizer, tweet):
    feature = tokenizer.texts_to_sequences([tweet])
    feature = pad_sequences(feature, maxlen=96)

    return feature


def get_features_tfidf(tfidf_vect, tweet):
    """
    Creates tweet features based on TF-IDF model
    """
    return tfidf_vect.transform(tweet).todense()


w2vec, cnn_model = load_word2vec_and_cnn_model()
tokenizer, lstm_model = load_vectorizer_and_lstm_model()
tfidf_nb, nb_clf = load_tfidf_and_clf_model(nb_path, nb_base_file_name)
tfidf_svc, svc_clf = load_tfidf_and_clf_model(svc_path, svc_base_file_name)


def get_prediction_for_tweet(tweet):
    """Creates predictions for a tweet using available models"""

    tweet = clean_tweet(tweet)
    tweet_words = tweet.split(" ")

    tweet_features_w2v = np.array([get_features_w2v(w2vec, tweet_words, cnn_model.get_input_shape_at(0)[1])])
    tweet_features_tokenizer = get_features_tokenizer(tokenizer, tweet)
    tweet_features_tfidf_nb = get_features_tfidf(tfidf_nb, [tweet])
    tweet_features_tfidf_svc = get_features_tfidf(tfidf_svc, [tweet])

    predictions = {
        "w2v_cnn": cnn_model.predict_classes(tweet_features_w2v).flatten()[0],
        "emb_cnn_lstm": lstm_model.predict_classes(tweet_features_tokenizer).flatten()[0],
        "tfidf_nb": nb_clf.predict(tweet_features_tfidf_nb)[0],
        "tfidf_svc": svc_clf.predict(tweet_features_tfidf_svc)[0]
    }

    return predictions
