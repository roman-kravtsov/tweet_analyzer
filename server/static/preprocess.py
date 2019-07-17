import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import re

nltk.download('stopwords')
nltk.download('wordnet')

negations_dic = {
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
    "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "isnt": "is not", "arent": "are not", "wasnt": "was not", "werent": "were not",
    "havent": "have not", "hasnt": "has not", "hadnt": "had not", "wont": "will not",
    "wouldnt": "would not", "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "cant": "can not", "couldnt": "could not", "shouldnt": "should not", "mightnt": "might not",
    "mustnt": "must not", "mightn": "might not", "mustn": "must not", "needn": "need not",
    "needn't": "need not", "shan": "shall not", "shan't": "shall not", "shant": "shall not"
}

stop_words = stopwords.words('english')
stop_words.append('im')
stop_words.append('u')

important_stop_words = list(negations_dic.keys())

for word in list(negations_dic.values()):
    important_stop_words.extend(list(word.split(" ")))

important_stop_words.append("not")
important_stop_words.append("no")


def filter_stop_words(stop_words, to_remove):
    correct_stop_words = list()
    for i in range(len(stop_words)):
        if not (stop_words[i] in to_remove):
            correct_stop_words.append(stop_words[i])

    return correct_stop_words


stop_words = filter_stop_words(stop_words, important_stop_words)


def clean_tweet(tweet):
    # Remove links
    tweet = re.sub('ht.?.?.?tps?://[A-Za-z0-9./]+', " ", tweet)

    # Remove mentioning @
    tweet = re.sub('@[A-Za-z0-9]+', " ", tweet)

    # Removing punctuation
    tweet = tweet.replace('[^\w\s]', '')

    # Split FancyWord into Fancy Word and tranform to lowercase
    tweet = " ".join([w.lower() for w in re.split('([A-Z][a-z]+)', tweet) if w])

    # Removing numerics and hashtags
    tweet = re.sub('[^a-zA-Z]', " ", tweet)

    # Remove rt
    tweet = re.compile(r'\brt\b').sub("", tweet)

    # Remove amp
    tweet = re.compile(r'\bamp\b').sub("", tweet)

    # Duplicates dropping
    tweet = " ".join([re.sub(r'(.)\1+', r'\1\1', tweet)])

    #  Transform negations into following form: wouldnt -> would not
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    tweet = neg_pattern.sub(lambda x: negations_dic[x.group()], tweet)

    # Removing stopwords
    tweet = ' '.join(word for word in tweet.split() if word not in stop_words)

    # Remove leftovers of http: ht... http... htt... and etc.
    tweet = re.sub(r'\bht..?t?.?p?s?.?\b', "", tweet)

    # Remove extra whitespaces
    tweet = ' '.join(tweet.split())

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])

    # Stemming
    stemmer = PorterStemmer()
    tweet = " ".join([stemmer.stem(word) for word in tweet.split()])

    if len(tweet) == 0:
        raise ValueError("Tweet empty after preprocessing, try another one")

    return tweet
