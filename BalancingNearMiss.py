import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate, train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from functions import model_predict, select_features, load_estimator, save_estimator, plot_feature_importance
from imblearn.under_sampling import NearMiss
import os
import matplotlib.pyplot as plt

# Read Dataset
print("[INFO] Reading dataset...")
#data = pd.read_csv("G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/TCGA-PANCAN-HiSeq-801x20531/data.csv", sep=',', header=0, index_col=0)
#labels = pd.read_csv("G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/TCGA-PANCAN-HiSeq-801x20531/labels.csv", sep=',', header=0, index_col=0)
data = pd.read_csv("C:/Users/Luigina/Il mio Drive/AI-Project/TCGA-PANCAN-HiSeq-801x20531/data.csv", sep=',', header=0, index_col=0)
labels = pd.read_csv("C:/Users\Luigina/Il mio Drive/AI-Project/TCGA-PANCAN-HiSeq-801x20531/labels.csv", sep=',', header=0, index_col=0)
print("[INFO] Finished reading dataset")

# Check Data
print("[INFO] Found NaN values in data: ", data.isna().values.any())
print("[INFO] Found NaN values in labels: ", labels.isna().values.any())

# Define Numpy Arrays
data_tonp = data.to_numpy()
labels_np = labels.to_numpy()
labels_np = np.ravel(labels_np)  # reshape labels along one direction

# NearMiss
nearmiss = NearMiss(version=3, n_jobs=-1)
data_resampled_np, labels_resampled_np = nearmiss.fit_resample(data, labels)
print("Total number of samples after NearMiss: ", len(data_resampled_np), ". Total number of labels ", len(labels_resampled_np))

# Feature Selection
vt = VarianceThreshold()
data_resampled_np = vt.fit_transform(data_resampled_np)
print("[INFO] Removed ", data_tonp.shape[1] - data_resampled_np.shape[1], " features with variance = 0 along samples")

# Get selected features NAMES
features = list(data.columns)
selected_features = list()
array = vt.get_support(True)
for i in array:
    selected_features.append(features[i])

# Scale the samples
print("[INFO] Scaling dataset...")
scaler = StandardScaler()
data_rs_sc = scaler.fit_transform(data_resampled_np)
print("[INFO] Finished scaling dataset")

# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
#make sure that the split is always the same,  and that the classes are somewhat balanced betweeen splits
data_train, data_test, labels_train, labels_test = train_test_split(data_rs_sc, np.ravel(labels_resampled_np), test_size = 0.30, random_state=12345, stratify = labels_resampled_np)
print("[INFO] Finished splitting dataset...")

# _______________________________________________________________Random Forest_______________________________________________________________________________________#
'''rdf_model=RandomForestClassifier(random_state=12345)
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
save_estimator(res_gridcv.best_estimator_, "RF_BD_NM.joblib")
print("[INFO] SVM_RFE model saved")

#predict
score = model_predict(model = res_gridcv.best_estimator_, test_data = data_test, test_labels = labels_test)
print("[RF_NM] Balanced accuracy score:", score)

'''
# Load Random Forest
best_rf = load_estimator("RF_BD_NM.joblib")

# evalute the best model on the test set
score = model_predict(model=best_rf, test_data=data_test, test_labels=labels_test)
print("[RANDOM FOREST] Balanced Accuracy score:", score)

# plot feature importances for the best model
plot_feature_importance(best_rf, "RF_BD_NM", selected_features)

# select important features based on threshold
imp_features, imp_features_test, feature_names_RFC = select_features(best_rf, 0.0004, data_train, data_test, True, selected_features)
print("[RANDOM FOREST] Found ", len(feature_names_RFC), " important features")


# _____________________________________________________________________SVM_RFE__________________________________________________________________________________#
#grid search with RFE
'''print("[INFO] Searching best params with GridSearchCV")
svm_model = LinearSVC(max_iter=10000)
rfe = RFECV(svm_model, step=10000, verbose=2)
pipe = Pipeline([('rfe', rfe), ('svm_model', svm_model)])
param_grid = {'svm_model__C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
              'svm_model__loss': ['hinge', 'squared_hinge']}
pipe_gridcv = GridSearchCV(pipe, param_grid=param_grid, cv=4, error_score='raise', n_jobs=-1, verbose=3, refit=True)
pipe_gridcv.fit(data_train, labels_train)
print(f"Best SVM model with params: {pipe_gridcv.best_params_} and score: {pipe_gridcv.best_score_:.3f}")
#Best SVM model with params: {'svm_model__C': 1e-05, 'svm_model__loss': 'hinge'} and score: 0.999
#save model

save_estimator(pipe_gridcv.best_estimator_, "SVM_RFE_BD.joblib")
print("[INFO] SVM_RFE model saved")
''''''
model = load_estimator("SVM_RFE_BD.joblib")
#evaluate performance on test set
score = model_predict(model = model, test_data = data_test, test_labels = labels_test)
print("[SVM_RFE] Balanced accuracy score:", score)

#get BEST features NAMES
mask = model.named_steps['rfe'].support_
feature_names_SVM_RFE = []
for i in range(len(mask)):
  if mask[i] == True:
    feature_names_SVM_RFE.append(selected_features[i])

print(len(feature_names_SVM_RFE), " important features\n", feature_names_SVM_RFE)

plot_feature_importance(estimator=model, name="SVM_RFE_BD.joblib", selected_features=selected_features)
'''