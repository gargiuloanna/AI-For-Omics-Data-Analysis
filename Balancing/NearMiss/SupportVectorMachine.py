# balancing
import numpy as np
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, remove_outliers, remove_correlated_features
from DatasetPrep.Scaling import scale
from DatasetPrep.VariablePreSelection import feature_pre_selection
from ModelEvaluation.Performance import balanced_model_predict, get_features_importances_SVM
from ModelEvaluation.SaveLoad import save_estimator

# _____________________________________________________________________READ DATASET_____________________________________________________________________#
# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
data, labels = remove_outliers(data, labels)
nearmiss = NearMiss(version=2, n_jobs=-1)
data_resampled_np, labels_resampled_np = nearmiss.fit_resample(data, labels)
print("Total number of samples after nearmiss: ", len(data_resampled_np), ". Total number of labels ", len(labels_resampled_np))
# Scale the samples
data_sc = scale(data_resampled_np)
# Feature Selection
data_sc = remove_correlated_features(data_sc, data.columns)
data_np, selected_features = feature_pre_selection(data, data_sc.to_numpy())

# _____________________________________________________________________SPLIT DATASET_____________________________________________________________________#
# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_np, np.ravel(labels_resampled_np), test_size=0.30, random_state=12345, stratify=labels_resampled_np)
print("[INFO] Finished splitting dataset...")

# _____________________________________________________________________SVM__________________________________________________________________________________#
print("[SVM WITH NEARMISS] Searching best params with GridSearchCV")

svm_model = LinearSVC(max_iter=10000, random_state=12345)
param_grid = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
              'loss': ['hinge', 'squared_hinge']}

svm_model_gridcv = GridSearchCV(svm_model, param_grid=param_grid, cv=4, scoring='accuracy', error_score='raise', n_jobs=-1, verbose=3, refit=True)
svm_model_gridcv.fit(data_train, labels_train)

print(f"[SVM WITH NEARMISS] Best SVM model with params: {svm_model_gridcv.best_params_} and score: {svm_model_gridcv.best_score_:.3f}")

# save model
svm_model = svm_model_gridcv.best_estimator_
save_estimator(svm_model, "SVM_NEARMISS.joblib")
print("[SVM  WITH NEARMISS] SVM_NEARMISS model saved")

# predict
score = balanced_model_predict(model=svm_model, name="SVM_NEARMISS", test_data=data_test, test_labels=labels_test)
print("[SVM WITH NEARMISS] Balanced accuracy score:", score)

# get important features per class
get_features_importances_SVM(svm=svm_model, selected_features=selected_features, name="SVM_NEARMISS")

