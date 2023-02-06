from sklearn.preprocessing import StandardScaler

def scale(data_np):
    print("[INFO] Scaling dataset...")
    scaler = StandardScaler()
    data_sc = scaler.fit_transform(data_np)
    print("[INFO] Finished scaling dataset")
    return data_sc