import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from functions import model_predict, select_features, save_estimator, plot_feature_importance

# Read Dataset
print("[INFO] Reading dataset...")
data = pd.read_csv("G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/TCGA-PANCAN-HiSeq-801x20531/data.csv", sep=',', header=0, index_col=0)
labels = pd.read_csv("G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/TCGA-PANCAN-HiSeq-801x20531/labels.csv", sep=',', header=0, index_col=0)
print("[INFO] Finished reading dataset")

# Check Data
print("[INFO] Found NaN values in data: ", data.isna().values.any())
print("[INFO] Found NaN values in labels: ", labels.isna().values.any())

# Define Numpy Arrays
data_tonp = data.to_numpy()
labels_np = labels.to_numpy()
labels_np = np.ravel(labels_np)  # reshape labels along one direction

# Feature Selection
vt = VarianceThreshold()
data_np = vt.fit_transform(data_tonp)
print("[INFO] Removed ", data_tonp.shape[1] - data_np.shape[1], " features with variance = 0 along samples")

# Get selected features NAMES
features = list(data.columns)
selected_features = list()
array = vt.get_support(True)
for i in array:
    selected_features.append(features[i])

# Scale the samples
print("[INFO] Scaling dataset...")
scaler = StandardScaler()
data_sc = scaler.fit_transform(data_np)
print("[INFO] Finished scaling dataset")

# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_sc, labels_np, test_size=0.30, random_state=12345, stratify=labels_np)
print("[INFO] Finished splitting dataset...")

rdf_model=RandomForestClassifier(random_state=12345)
param_grid = {
    'n_estimators': [30, 40, 50, 60, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [0.1, 0.23, 2, 10, 20],
    'min_samples_leaf': [0.1, 0.23, 1, 10, 20],
    'max_features': ['sqrt', 'log2']
}
res_gridcv=GridSearchCV(rdf_model, param_grid=param_grid, cv=4, error_score='raise', n_jobs=-1, verbose=3, refit=True)
res_gridcv.fit(data_train, labels_train)
print(f"Best random forest with params: {res_gridcv.best_params_} and score: {res_gridcv.best_score_:.3f}")

#save model
save_estimator(res_gridcv.best_estimator_, "RF_NB.joblib")
print("[INFO] RF_NB model saved")

#predict
score = model_predict(model = res_gridcv.best_estimator_, test_data = data_test, test_labels = labels_test)
print("[RANDOM_FOREST] Balanced accuracy score:", score)

# plot feature importances for the best model
plot_feature_importance(res_gridcv, "RF_NB", selected_features)

# select important features based on threshold
imp_features, imp_features_test, feature_names_RFC = select_features(res_gridcv, 0.0004, data_train, data_test, True, selected_features)
print("[RANDOM FOREST] Found ", len(feature_names_RFC), " important features")

