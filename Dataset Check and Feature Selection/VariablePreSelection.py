from sklearn.feature_selection import VarianceThreshold



def feature_pre_selection(data, threshold = 0):
    vt = VarianceThreshold()
    data_np = vt.fit_transform(data)
    print("[INFO] Removed ", data.shape[1] - data_np.shape[1], " features with variance = 0 along samples")
    return data_np


def get_selected_feature_names(data, vt):
    selected_features = list()
    features = list(data.columns)
    array = vt.get_support(True)
    for i in array:
        selected_features.append(features[i])
        return selected_features