import pandas as pd
from sklearn.utils import shuffle


def get_data(path, train_size, test_size, headers, skip_rows=0):
    """ Loads data from the specified path with given training and test split sizes"""

    data = pd.read_csv(path,
                       names=headers,
                       low_memory=False,
                       header=None,
                       encoding="ISO-8859-1",
                       skiprows=skip_rows)

    data = shuffle(data, random_state=9016832).reset_index(drop=True)

    if test_size + train_size > len(data):
        raise ValueError("Train and test datasets will overlap, please adjust train and test set sizes")

    x_test = pd.DataFrame(data['text']).truncate(before=(data.shape[0] - test_size)).reset_index(drop=True)
    y_test = pd.DataFrame(data['polarity']).truncate(before=(data.shape[0] - test_size)).reset_index(drop=True)

    x_train = pd.DataFrame(data['text']).truncate(after=(train_size - 1)).reset_index(drop=True)
    y_train = pd.DataFrame(data['polarity']).truncate(after=(train_size - 1)).reset_index(drop=True)

    del data

    return x_test, y_test, x_train, y_train
