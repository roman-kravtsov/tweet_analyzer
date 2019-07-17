import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from classification.util.features_extractor import update_labels
from classification.util.data_util import get_data
from classification.util.plot import plot_confusion_matrix

save_path_prefix = "models_clf/"

"""Data preparation"""
data_path = r"D:\DOCS\University of Passau\Text Mining\data\16kk\train_data.csv"
data_headers = ["polarity", "id", "date", "query", "user", "text"]

train_size = 10000
test_size = train_size * 0.2

x_test, y_test, x_train, y_train = get_data(data_path, train_size, test_size, data_headers)
train_labels, test_labels = update_labels(y_train, y_test)

"""TFiDF"""
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 1))
vectorizer = tfidf.fit(x_train["text"])

features_train = pd.DataFrame(vectorizer.transform(x_train["text"]).todense(), columns=tfidf.get_feature_names())
features_test = pd.DataFrame(vectorizer.transform(x_test["text"]).todense(), columns=tfidf.get_feature_names())

"""Support Vector Machine Classifier"""
clf = SVC(kernel='linear').fit(features_train.values, train_labels)
predicted = clf.predict(features_test.values)

print(metrics.classification_report(test_labels, predicted, target_names=["negative", "positive"]))
print(metrics.confusion_matrix(test_labels, predicted))

plot_confusion_matrix(test_labels, predicted, [0, 1], ["Negative", "Positive"], save_path_prefix + "svc_raw_cf_10k_2k",
                      normalize=True)

filename = save_path_prefix + "svc_raw_data_10k_train_2k_test.clf"
pickle.dump(clf, open(filename, 'wb'))
