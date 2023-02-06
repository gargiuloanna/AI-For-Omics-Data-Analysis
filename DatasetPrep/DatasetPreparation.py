import numpy as np
import pandas as pd


def read_dataset(directory):
    print("[INFO] Reading dataset...")
    data = pd.read_csv(directory + "TCGA-PANCAN-HiSeq-801x20531/data.csv", sep=',', header=0, index_col=0)
    labels = pd.read_csv(directory + "TCGA-PANCAN-HiSeq-801x20531/labels.csv", sep=',', header=0, index_col=0)
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


def get_dataset(directory):
    data, labels = read_dataset(directory)
    if not check_dataset(data, labels):
        return dataframe_to_numpy
    else:
        print("Check for NaN values returned True")
        return None, None
