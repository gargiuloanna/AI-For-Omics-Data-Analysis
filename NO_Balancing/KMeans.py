#no balancing

import pandas as pd
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy, outliers
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from ModelEvaluation.SaveLoad import save_estimator
from sklearn.decomposition import PCA
from ModelEvaluation.Performance import plot_clustering
from sklearn.cluster import KMeans

#--------------------------------------------------------------------READ DATASET--------------------------------------------------------------------#
# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
#outliers(data, data.columns, replace=True)
data_np, labels_np = dataframe_to_numpy(data, labels)
# Feature Selection
data_np, selected_features = feature_pre_selection(data)

# Scale the samples
data_sc = scale(data_np)
#PCA
pca = PCA(n_components=5)
df = pca.fit_transform(data_np)
print(pca.explained_variance_)
df_pd = pd.DataFrame(data = df, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
codes = {'BRCA':'red', 'COAD':'green', 'LUAD':'blue', 'PRAD':'violet', 'KIRC':'orange'}
pd.plotting.scatter_matrix(df_pd, c=labels.Class.map(codes), figsize = (30, 30));
df_pd.drop(['principal component 2','principal component 4','principal component 5'], axis=1, inplace=True)
df = df_pd.to_numpy()

# _____________________________________________________________________K MEANS__________________________________________________________________________________#
clusterer = KMeans(n_clusters=5, random_state=12345, n_init = 100, algorithm= 'elkan')
cluster_labels = clusterer.fit_predict(df)
plot_clustering(clusterer=clusterer, cluster_labels=cluster_labels, n_clusters=5, df=df)

#save_estimator
save_estimator(directory, clusterer, "KMeans.joblib")