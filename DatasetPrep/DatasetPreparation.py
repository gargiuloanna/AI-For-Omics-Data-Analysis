import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

directory = "G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project"


def read_dataset():
    print("[INFO] Reading dataset...")
    data = pd.read_csv(directory + "/TCGA-PANCAN-HiSeq-801x20531/data.csv", sep=',', header=0, index_col=0)
    labels = pd.read_csv(directory + "/TCGA-PANCAN-HiSeq-801x20531/labels.csv", sep=',', header=0, index_col=0)
    print("[INFO] Finished reading dataset")
    return data, labels


def check_dataset(data, labels):
    data_na = data.isna().values.any()
    labels_na = labels.isna().values.any()
    print("[INFO] Found NaN values in data: ", data_na)
    print("[INFO] Found NaN values in labels: ", labels_na)
    return data_na, labels_na


def dataframe_to_numpy(data, labels):
    data_np = data.to_numpy()
    labels_np = labels.to_numpy()
    labels_np = np.ravel(labels_np)  # reshape labels along one direction
    return data_np, labels_np


def get_dataset():
    data, labels = read_dataset()
    if not check_dataset(data, labels):
        return dataframe_to_numpy
    else:
        print("Check for NaN values returned True")
        return None, None


def remove_outliers(data, labels):
    local = LocalOutlierFactor(n_neighbors=15, n_jobs=-1)
    outliers = local.fit_predict(data)
    print("[INFO] Removed ", data[outliers==-1].shape[0], " outliers")
    return data[outliers == 1], labels[outliers == 1]


def remove_correlated_features(data, columns_names):
    cov = np.cov(data, rowvar=False)
    c = pd.DataFrame(np.abs(cov), columns=columns_names)
    # select upper traingle of correlation matrix
    upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    data_sc = pd.DataFrame(data, columns=columns_names)
    data_sc.drop(to_drop, axis=1, inplace=True)
    print("[INFO] Removed ", len(to_drop), "correlated features")
    return data_sc