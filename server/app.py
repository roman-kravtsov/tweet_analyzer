from flask import Flask
from flask import request, jsonify, make_response
from flask_cors import CORS

from static.prediction import get_prediction_for_tweet

import tensorflow as tf

app = Flask(__name__)
CORS(app)
graph = tf.get_default_graph()


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.get_json()['tweet']

    # required to correctly handle tensorflow session
    global graph
    with graph.as_default():
        predictions = get_prediction_for_tweet(tweet)

    response_body = {
        "w2v_cnn": str(predictions["w2v_cnn"]),
        "emb_cnn_lstm": str(predictions["emb_cnn_lstm"]),
        "tfidf_nb": str(predictions["tfidf_nb"]),
        "tfidf_svc": str(predictions["tfidf_svc"]),
    }

    return make_response(jsonify(response_body), 200)


if __name__ == '__main__':
    app.run()
