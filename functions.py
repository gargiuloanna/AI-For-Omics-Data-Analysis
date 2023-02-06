import os

import matplotlib.pyplot as plt
import scikitplot as skplt
import scikitplot.metrics as skplt_m
from joblib import dump, load
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score

# define directory to save & load the models
directory = 'G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/Models/'


def model_predict(model, test_data, test_labels):
    lab_pred = model.predict(test_data)
    score = balanced_accuracy_score(test_labels, lab_pred)
    skplt_m.plot_confusion_matrix(test_labels, lab_pred)
    return score


def select_features(model, threshold, train, test, prefit, selected_features):
    sfm = SelectFromModel(estimator=model, threshold=threshold, prefit=prefit)
    imp_features = sfm.transform(train)
    imp_features_test = sfm.transform(test)
    feature_names = sfm.get_feature_names_out(selected_features)
    return imp_features, imp_features_test, feature_names


def plot_feature_importance(estimator, name, selected_features):
    skplt.estimators.plot_feature_importances(estimator, feature_names=selected_features, max_num_features=300, figsize=(100, 100))
    plt.savefig(name + "_PLOT.png")
    plt.show()


def save_estimator(estimator, name):
    filename = os.path.join(directory, name)
    dump(estimator, filename)


def load_estimator(name):
    filename = os.path.join(directory, name)
    estimator = load(filename)
    return estimator
