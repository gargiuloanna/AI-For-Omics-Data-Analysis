from sklearn.preprocessing import StandardScaler


def scale(data_np, with_mean=True):
    print("[INFO] Scaling dataset...")
    scaler = StandardScaler(with_mean=with_mean)
    data_sc = scaler.fit_transform(data_np)
    print("[INFO] Finished scaling dataset")
    return data_sc