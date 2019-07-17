import pickle


def save_classifier(clf, vect, filename_clf, filename_vect):
    pickle.dump(clf, open(filename_clf, 'wb'))
    pickle.dump(vect, open(filename_vect, 'wb'))
