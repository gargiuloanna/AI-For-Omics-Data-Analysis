import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scikitplot as skplt
import scikitplot.metrics as skplt_m
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score, accuracy_score, silhouette_samples, silhouette_score

directory = "G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project"

def unbalanced_model_predict(model, name, test_data, test_labels):
    lab_pred = model.predict(test_data)
    score = balanced_accuracy_score(test_labels, lab_pred)
    skplt_m.plot_confusion_matrix(test_labels, lab_pred)
    plt.savefig(directory + "/Plots/" + name + "_CONFUSION.png")
    return score

def balanced_model_predict(model, name, test_data, test_labels):
    lab_pred = model.predict(test_data)
    score = accuracy_score(test_labels, lab_pred)
    skplt_m.plot_confusion_matrix(test_labels, lab_pred)
    plt.savefig(directory + "/Plots/" + name + "_CONFUSION.png")
    return score


def select_features_from_model(model, threshold, prefit, selected_features, train = None, test = None):
    sfm = SelectFromModel(estimator=model, threshold=threshold, prefit=prefit)
    imp_features = sfm.transform(train)
    imp_features_test = sfm.transform(test)
    feature_names = sfm.get_feature_names_out(selected_features)
    return imp_features, imp_features_test, feature_names


def plot_feature_importance(estimator, name, selected_features):
    skplt.estimators.plot_feature_importances(estimator, feature_names=selected_features, max_num_features=300, figsize=(100, 100))
    filename = os.path.join(directory + "/Plots/", name + "_PLOT.png")
    plt.savefig(filename)
    plt.show()

def plot_clustering(clusterer, cluster_labels, n_clusters, df):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(df, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        df[:, 0], df[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = 5",
        fontsize=14,
        fontweight="bold",
    )

    plt.show()
    plt.savefig(directory + "/Plots/KMeans.png")

