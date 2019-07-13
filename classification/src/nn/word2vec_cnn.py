from classification.src.util.preprocessor import preprocess
from classification.src.util.features_extractor import get_word2vec_features, update_labels
from classification.src.util.data_util import get_data
from classification.src.util.model import get_model_word2vec_cnn, plot_training_results

from sklearn.metrics import confusion_matrix

"""Data preparation"""
data_path = r"D:\DOCS\University of Passau\Text Mining\data\16kk\train_data.csv"

train_size = 500
batch_size = 50

x_test, y_test, x_train, y_train = get_data(data_path, train_size)

x_test, y_test = preprocess(x_test, y_test)
x_train, y_train = preprocess(x_test, y_test)

features_train, features_test = get_word2vec_features(x_train, x_test)
train_labels, test_labels = update_labels(y_train, y_test)


"""Training and evaluation"""
model = get_model_word2vec_cnn(features_train[0].shape, 1000, 10)

history = model.fit(features_train, train_labels, validation_data=(features_test, test_labels),
                    epochs=30, batch_size=100)
# model.summary()

# Validation and training loss chart
plot_training_results(history)

# Confusion matrix
y_pred = model.predict_classes(features_test)
matrix = confusion_matrix(test_labels, y_pred, labels=[0, 1])

print(matrix)

# Print incorrectly classified
# for i in range(len(x_test["text"])):
#     if test_labels_updated[i] != y_pred[i]:
#         print(i, test_labels_updated[i], y_pred[i], x_test.at[i, "text"])
