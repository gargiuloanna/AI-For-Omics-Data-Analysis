# no balancing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy, remove_outliers
from DatasetPrep.VariablePreSelection import feature_pre_selection
from ModelEvaluation.Performance import plot_clustering, get_features_PCA
from ModelEvaluation.SaveLoad import save_estimator

# _____________________________________________________________________READ DATASET_____________________________________________________________________#

# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
data, labels = remove_outliers(data, labels)
data_np, labels_np = dataframe_to_numpy(data, labels)
# Feature Selection
data_np, selected_features = feature_pre_selection(data, data_np)

# _____________________________________________________________________PCA_____________________________________________________________________#

pca = PCA(n_components=10)
df = pca.fit_transform(data_np)

variance = pca.explained_variance_ratio_ * 100

for v in variance:
    print(f"% Variance Ratio per PC ", v)

codes = {'BRCA': 'red', 'COAD': 'green', 'LUAD': 'blue', 'PRAD': 'violet', 'KIRC': 'orange'}
df_pd = pd.DataFrame(data=df, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8', 'PC 9', 'PC 10'])
pd.plotting.scatter_matrix(df_pd, c=labels.Class.map(codes), figsize=(50, 50))
df_pd.drop(['PC 2', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8', 'PC 9', 'PC 10'], axis=1, inplace=True)
df = df_pd.to_numpy()

# _____________________________________________________________________Feature Names_____________________________________________________________________#

get_features_PCA(selected_features, pca.components_[0], "component1")
get_features_PCA(selected_features, pca.components_[2], "component3")


# _____________________________________________________________________K MEANS__________________________________________________________________________________#

clusterer = KMeans(n_clusters=5, random_state=12345, n_init=100, algorithm='elkan')
cluster_labels = clusterer.fit_predict(df)
plot_clustering(clusterer=clusterer, cluster_labels=cluster_labels, n_clusters=5, df=df)

# save_estimator
save_estimator(clusterer, "KMeans.joblib")
