import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from classification.src.util.data_util import get_data
from classification.src.util.features_extractor import update_labels
from classification.src.util.plot import plot_confusion_matrix
from classification.src.util.preprocessor import preprocess
from classification.src.util.save_util import save_classifier

save_path_prefix = "models_clf/"

"""Data preparation"""
data_path = r"D:\DOCS\University of Passau\Text Mining\data\16kk\train_data.csv"
data_headers = ["polarity", "id", "date", "query", "user", "text"]

train_size = 50000
test_size = train_size * 0.2

x_test, y_test, x_train, y_train = get_data(data_path, train_size, test_size, data_headers)

x_test, y_test = preprocess(x_test, y_test)
x_train, y_train = preprocess(x_train, y_train)

train_labels, test_labels = update_labels(y_train, y_test)

"""TFiDF features"""
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 1))
vectorizer = tfidf.fit(x_train["text"])

features_train = pd.DataFrame(vectorizer.transform(x_train["text"]).todense(), columns=tfidf.get_feature_names())
features_test = pd.DataFrame(vectorizer.transform(x_test["text"]).todense(), columns=tfidf.get_feature_names())

"""Naive Bayes Classifier"""
clf = MultinomialNB().fit(features_train.values, train_labels)
predicted = clf.predict(features_test.values)

print(metrics.classification_report(test_labels, predicted, target_names=["negative", "positive"]))
print(metrics.confusion_matrix(test_labels, predicted))

plot_confusion_matrix(test_labels, predicted, [0, 1], ["Negative", "Positive"], save_path_prefix + "nb_prep_cf_50k_10k",
                      normalize=True)

filename_clf = save_path_prefix + "nb_prep_data_50k_train_10k_test.clf"
filename_vect = save_path_prefix + "nb_prep_data_50k_train_10k_test.vect"

save_classifier(clf, vectorizer, filename_clf, filename_vect)
