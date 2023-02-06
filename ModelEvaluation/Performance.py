import os
import matplotlib.pyplot as plt
import scikitplot as skplt
import scikitplot.metrics as skplt_m
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score


def model_predict(model, name, test_data, test_labels, directory):
    lab_pred = model.predict(test_data)
    score = balanced_accuracy_score(test_labels, lab_pred)
    skplt_m.plot_confusion_matrix(test_labels, lab_pred)
    plt.savefig(directory + "/Plots/" + name + "_CONFUSION.png")
    return score


def select_features_from_model(model, threshold, prefit, selected_features, train = None, test = None):
    sfm = SelectFromModel(estimator=model, threshold=threshold, prefit=prefit)
    imp_features = sfm.transform(train)
    imp_features_test = sfm.transform(test)
    feature_names = sfm.get_feature_names_out(selected_features)
    return imp_features, imp_features_test, feature_names


def plot_feature_importance(estimator, name, selected_features, directory):
    skplt.estimators.plot_feature_importances(estimator, feature_names=selected_features, max_num_features=300, figsize=(100, 100))
    filename = os.path.join(directory + "/Plots/", name + "_PLOT.png")
    plt.savefig(filename)
    plt.show()
