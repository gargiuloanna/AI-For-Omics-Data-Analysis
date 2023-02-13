import os

from joblib import dump, load

directory = "G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project"


def save_estimator(estimator, name):
    filename = os.path.join(directory + "/Models/", name)
    dump(estimator, filename)


def load_estimator(name):
    filename = os.path.join(directory + "/Models/", name)
    estimator = load(filename)
    return estimator
