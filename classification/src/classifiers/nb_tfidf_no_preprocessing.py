import pandas as pd
from classification.src.util.data_util import get_data

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

"""Data preparation"""
data_path = r"D:\DOCS\University of Passau\Text Mining\data\16kk\train_data.csv"
train_size = 50000

x_test, y_test, x_train, y_train = get_data(data_path, train_size)

"""TFiDF features"""
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 1))
vectorizer = tfidf.fit(x_train["text"])

features_train = pd.DataFrame(vectorizer.transform(x_train["text"]).todense(), columns=tfidf.get_feature_names())
features_test = pd.DataFrame(vectorizer.transform(x_test["text"]).todense(), columns=tfidf.get_feature_names())

"""Naive Bayes Classifier"""
clf = MultinomialNB().fit(features_train.values, y_train["polarity"])
predicted = clf.predict(features_test.values)

print(metrics.classification_report(y_test["polarity"], predicted, target_names=["negative", "positive"]))
print(metrics.confusion_matrix(y_test["polarity"], predicted))
