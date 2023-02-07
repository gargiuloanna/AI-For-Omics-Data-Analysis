#no balancing

from sklearn.model_selection import train_test_split
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from ModelEvaluation.SaveLoad import save_estimator
from sklearn.decomposition import PCA
from ModelEvaluation.Performance import model_predict, select_features_from_model, plot_feature_importance
from sklearn.cluster import KMeans
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#--------------------------------------------------------------------READ DATASET--------------------------------------------------------------------#
directory = "G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/"
#directory = "C:/Users/Luigina/Il mio Drive/AI-Project/"

# Read & Check dataset
data, labels = read_dataset(directory)
check_dataset(data, labels)
def determine_outlier_thresholds_std(dataframe, col_name):
    upper_boundary = dataframe[col_name].mean() + 3 * dataframe[col_name].std()
    lower_boundary = dataframe[col_name].mean() - 3 * dataframe[col_name].std()
    return lower_boundary, upper_boundary

def check_outliers_std(dataframe, col_name):
    lower_boundary, upper_boundary = determine_outlier_thresholds_std(dataframe, col_name)
    if dataframe[(dataframe[col_name] > upper_boundary) | (dataframe[col_name] < lower_boundary)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds_std(dataframe, cols, replace=False):
    from tabulate import tabulate
    data = []
    for col_name in cols:
        if col_name != 'Outcome':
            outliers_ = check_outliers_std(dataframe, col_name)
            count = None
            lower_limit, upper_limit = determine_outlier_thresholds_std(dataframe, col_name)
            if outliers_:
                count = dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)][col_name].count()
                if replace:
                    if lower_limit < 0:
                        # We don't want to replace with negative values, right!
                        dataframe.loc[(dataframe[col_name] > upper_limit), col_name] = upper_limit
                    else:
                        dataframe.loc[(dataframe[col_name] < lower_limit), col_name] = lower_limit
                        dataframe.loc[(dataframe[col_name] > upper_limit), col_name] = upper_limit
            outliers_status = check_outliers_std(dataframe, col_name)
            data.append([outliers_, outliers_status,count, col_name, lower_limit, upper_limit])
    table = tabulate(data, headers=['Outlier (Previously)','Outliers','Count', 'Column','Lower Limit', 'Upper Limit'], tablefmt='rst', numalign='right')
    print("Removing Outliers using 3 Standard Deviation")
    print(table)


data_np, labels_np = dataframe_to_numpy(data, labels)

# Feature Selection
data_np, selected_features = feature_pre_selection(data)

# Scale the samples
data_sc = scale(data_np)

np = pd.DataFrame(data_sc, columns=selected_features)
cp = np.copy()
replace_with_thresholds_std(cp, cp.columns, replace=True)

df_diff = pd.concat([np, cp]).drop_duplicates(keep=False)
print(df_diff)

#PCA
pca = PCA(n_components=2)
df = pca.fit_transform(data_np)
#df_pd = pd.DataFrame(data = df, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
codes = {'BRCA':'red', 'COAD':'green', 'LUAD':'blue', 'PRAD':'violet', 'KIRC':'orange'}
#pd.plotting.scatter_matrix(df_pd, c=labels.Class.map(codes), figsize = (30, 30));
#print(df_pd.head())
#df_pd.drop(['principal component 2','principal component 4','principal component 5'], axis=1, inplace=True)
#print(df_pd.head())
'''
# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_sc, labels_np, test_size=0.30, random_state=12345, stratify=labels_np)
print("[INFO] Finished splitting dataset...")
'''
# _____________________________________________________________________K MEANS__________________________________________________________________________________#
cluster = KMeans(n_clusters=5, random_state = 12345)
cluster_labels = cluster.fit_predict(df)
save_estimator(directory, cluster, "k_means.joblib")
# Getting unique labels
u_labels = np.unique(cluster_labels)
# plotting the results:
for i in u_labels:
    plt.scatter(df[cluster_labels == i, 0], df[cluster_labels == i, 1], label=i)
plt.legend()
skplt.metrics.plot_silhouette(data_np, cluster_labels)
plt.show()