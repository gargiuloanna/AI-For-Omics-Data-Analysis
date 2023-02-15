from sklearn.feature_selection import VarianceThreshold


def feature_pre_selection(data, data_np, threshold=0):
    vt = VarianceThreshold()
    data_removed = vt.fit_transform(data_np)
    print("[INFO] Removed ", data_np.shape[1] - data_removed.shape[1],
          " features with variance = ", threshold, " along samples")
    selected_features = list()
    features = list(data.columns)
    array = vt.get_support(True)
    for i in array:
        selected_features.append(features[i])

    return data_removed, selected_features
