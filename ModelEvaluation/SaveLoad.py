import os
from joblib import dump, load


def save_estimator(directory, estimator, name):
    filename = os.path.join(directory + "/Models/", name)
    dump(estimator, filename)


def load_estimator(directory, name):
    filename = os.path.join(directory + "/Models/", name)
    estimator = load(filename)
    return estimator
