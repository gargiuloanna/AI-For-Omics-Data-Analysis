import numpy as np
import pandas as pd
from tabulate import tabulate


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

def outliers(dataframe, cols, replace = False):
    data = []
    for col_name in cols:
        if col_name != 'Outcome':
            outliers_ = _check_outliers_std(dataframe, col_name)
            count = None
            lower_limit, upper_limit = _determine_outlier_thresholds_std(dataframe, col_name)
            if outliers_:
                count = dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)][col_name].count()
                if replace:
                    if lower_limit < 0:
                        # We don't want to replace with negative values, right!
                        dataframe.loc[(dataframe[col_name] > upper_limit), col_name] = upper_limit
                    else:
                        dataframe.loc[(dataframe[col_name] < lower_limit), col_name] = lower_limit
                        dataframe.loc[(dataframe[col_name] > upper_limit), col_name] = upper_limit
            outliers_status = _check_outliers_std(dataframe, col_name)
            data.append([outliers_, outliers_status, count, col_name, lower_limit, upper_limit])
    table = tabulate(data, headers=['Outlier (Previously)', 'Outliers', 'Count', 'Column', 'Lower Limit', 'Upper Limit'], tablefmt='rst', numalign='right')
    print("Removing Outliers using 3 Standard Deviation")
    print(table)

def _determine_outlier_thresholds_std(dataframe, col_name):
    upper_boundary = dataframe[col_name].mean() + 3 * dataframe[col_name].std()
    lower_boundary = dataframe[col_name].mean() - 3 * dataframe[col_name].std()
    return lower_boundary, upper_boundary

def _check_outliers_std(dataframe, col_name):
    lower_boundary, upper_boundary = _determine_outlier_thresholds_std(dataframe, col_name)
    if dataframe[(dataframe[col_name] > upper_boundary) | (dataframe[col_name] < lower_boundary)].any(axis=None):
        return True
    else:
        return False