import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from wordcloud import WordCloud

from sklearn.metrics import confusion_matrix


def plot_training_results(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_confusion_matrix(y_test,
                          predictions,
                          classes,
                          labels,
                          filepath,
                          normalize=False,
                          cmap=plt.cm.YlOrRd):
    """Plots and saves confusion matrix"""
    plt.figure()
    plt.autoscale(False)
    plt.gca().invert_yaxis()

    cm = confusion_matrix(y_test, predictions, labels=classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    plt.xlim((-0.5, len(classes) - 0.5))
    plt.ylim((len(classes) - 0.5, -0.5))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig("{}.png".format(filepath), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_word_cloud(tweets, polarity = 0):
  word_string = pd.Series(tweets[tweets['polarity'] == polarity].text.values).str.cat(sep=' ')
  wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(word_string)
  plt.figure(figsize=(12,10))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.show()  

def count_word_freq(tweets):
  freq_list = {}
  
  for tweet in tweets:
    for word in tweet.split():
      if word in freq_list:
        freq_list[word] = freq_list[word] + 1
      else:
        freq_list[word] = 1
  
  word_freq = pd.Series(freq_list)
  
  return word_freq.sort_values(ascending = False)

def plot_n_freq_words(tweets, top_number = 20):
    word_freq = count_word_freq(tweets.text.values)
    top_words = word_freq.head(top_number)
    top_value = word_freq.head(20)[:1][0]
    bottom_value = word_freq.head(20)[-1:][0]
    top_words.plot.bar(figsize=(15,10), xlim= (bottom_value, top_value))