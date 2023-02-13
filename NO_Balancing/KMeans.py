#no balancing

import numpy as np
import pandas as pd
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy, remove_outliers
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from ModelEvaluation.SaveLoad import save_estimator
from sklearn.decomposition import PCA
from ModelEvaluation.Performance import plot_clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#_____________________________________________________________________READ DATASET_____________________________________________________________________#

# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
data, labels = remove_outliers(data, labels)
data_np, labels_np = dataframe_to_numpy(data, labels)
# Scale the samples
data_sc = scale(data_np, with_mean=False)
# Feature Selection
data_np, selected_features = feature_pre_selection(data, data_sc)


#_____________________________________________________________________PCA_____________________________________________________________________#

pca = PCA(n_components=10)
df = pca.fit_transform(data_np)

variance = pca.explained_variance_ratio_ * 100

for v in variance:
    print(f"% Variance Ratio per PC ", v)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


codes = {'BRCA':'red', 'COAD':'green', 'LUAD':'blue', 'PRAD':'violet', 'KIRC':'orange'}
df_pd = pd.DataFrame(data = df, columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8', 'PC 9', 'PC 10'])
pd.plotting.scatter_matrix(df_pd, c=labels.Class.map(codes), figsize = (50, 50))
df_pd.drop(['PC 2', 'PC 4','PC 5','PC 6','PC 7','PC 8','PC 9','PC 10'], axis=1, inplace=True)
df = df_pd.to_numpy()

#_____________________________________________________________________Feature Names_____________________________________________________________________#

n_pcs= pca.components_.shape[0]
most_important =[np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
most_important_names=[selected_features[most_important[i]] for i in range(n_pcs)]
print(most_important_names)

# _____________________________________________________________________K MEANS__________________________________________________________________________________#

clusterer = KMeans(n_clusters=5, random_state=12345, n_init = 100, algorithm= 'elkan')
cluster_labels = clusterer.fit_predict(df)
plot_clustering(clusterer=clusterer, cluster_labels=cluster_labels, n_clusters=5, df=df)

#save_estimator
save_estimator(clusterer, "KMeans.joblib")