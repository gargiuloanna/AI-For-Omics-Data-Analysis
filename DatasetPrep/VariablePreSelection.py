from sklearn.feature_selection import VarianceThreshold


def feature_pre_selection(data, data_np = None, threshold=0):
    if data_np == None:
        data_np = data
    vt = VarianceThreshold()
    data_np = vt.fit_transform(data_np)
    print("[INFO] Removed ", data.shape[1] - data_np.shape[1], " features with variance = ", threshold, " along samples")
    selected_features = list()
    features = list(data.columns)
    array = vt.get_support(True)
    for i in array:
        selected_features.append(features[i])

    return data_np, selected_features
