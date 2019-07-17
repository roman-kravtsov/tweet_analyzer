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
stop_words.remove('not')
stop_words.remove('no')


def clean_data(tweets, inplace=False):
    data = tweets.copy(deep=True) if not inplace else tweets

    # Remove links
    data['text'] = data['text'].apply(lambda text: re.sub('ht.?.?.?tps?://[A-Za-z0-9./]+', " ", text))

    # Remove mentioning @
    data['text'] = data['text'].apply(lambda text: re.sub('@[A-Za-z0-9]+', " ", text))

    # Removing punctuation
    data['text'] = data['text'].str.replace('[^\w\s]', '')

    # Split FancyWord into Fancy Word and tranform to lowercase
    data['text'] = data['text'].apply(lambda text: " ".join([w.lower() for w in re.split('([A-Z][a-z]+)', text) if w]))

    # Removing numerics and hashtags
    data['text'] = data['text'].apply(lambda text: re.sub('[^a-zA-Z]', " ", text))

    # Remove rt
    data['text'] = data['text'].apply(lambda text: re.compile(r'\brt\b').sub("", text))

    # Remove amp
    data['text'] = data['text'].apply(lambda text: re.compile(r'\bamp\b').sub("", text))

    # Duplicates dropping
    data['text'] = data['text'].apply(lambda text: " ".join([re.sub(r'(.)\1+', r'\1\1', text)]))

    # Transform negations into following form: wouldnt -> would not
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    data['text'] = data['text'].apply(lambda text: neg_pattern.sub(lambda x: negations_dic[x.group()], text))

    # Removing stopwords
    data['text'] = data['text'].apply(lambda text: ' '.join(word for word in text.split() if word not in stop_words))

    # Remove leftovers of http: ht... http... htt... and etc.
    data['text'] = data['text'].apply(lambda text: re.sub(r'\bht..?t?.?p?s?.?\b', "", text))

    # Remove extra whitespaces
    data['text'] = data['text'].apply(lambda text: ' '.join(text.split()))

    return data


def stem(tweets, inplace=False):
    data = tweets.copy(deep=True) if not inplace else tweets

    # Stemming
    stemmer = PorterStemmer()
    data['text'] = data['text'].apply(lambda text: " ".join([stemmer.stem(word) for word in text.split()]))

    return data


def lem(tweets, inplace=False):
    data = tweets.copy(deep=True) if not inplace else tweets

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    data['text'] = data['text'].apply(lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()]))

    return data


# remove empty tweets
def drop_empty(x, y):
    to_drop = []

    for i, t in enumerate(x["text"]):
        if len(t) == 0:
            to_drop.append(i)

    x.drop(to_drop, inplace=True)
    y.drop(to_drop, inplace=True)


def preprocess(x, y):
    x = clean_data(x)
    drop_empty(x, y)
    x = lem(x)
    x = stem(x)

    return x, y
