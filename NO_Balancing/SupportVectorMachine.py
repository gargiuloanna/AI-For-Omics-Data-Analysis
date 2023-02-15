# no balancing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy, remove_outliers, \
    remove_correlated_features
from DatasetPrep.Scaling import scale
from DatasetPrep.VariablePreSelection import feature_pre_selection
from ModelEvaluation.Performance import unbalanced_model_predict, get_features_importances_SVMRFE
from ModelEvaluation.SaveLoad import save_estimator

# _____________________________________________________________________READ DATASET_____________________________________________________________________#
# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
data, labels = remove_outliers(data, labels)
data_np, labels_np = dataframe_to_numpy(data, labels)
# Scale the samples
data_sc = scale(data_np)
# Feature Selection
data_sc = remove_correlated_features(data_sc, data.columns)
data_np, selected_features = feature_pre_selection(data, data_sc.to_numpy())

# _____________________________________________________________________SPLIT DATASET_____________________________________________________________________#
# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_np, labels_np, test_size=0.30, random_state=12345, stratify=labels_np)
print("[INFO] Finished splitting dataset...")

# _____________________________________________________________________SVM_RFE__________________________________________________________________________________#
# grid search with RFE
print("[SVM_RFE] Searching best params with GridSearchCV")

svm_model = LinearSVC(max_iter=5000, random_state=12345)
rfe = RFECV(svm_model, step=9000, verbose=2)
pipe = Pipeline([('rfe', rfe), ('svm_model', svm_model)])
param_grid = {'svm_model__C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
              'svm_model__loss': ['hinge', 'squared_hinge']}

pipe_gridcv = GridSearchCV(pipe, param_grid=param_grid, cv=4, scoring='balanced_accuracy', error_score='raise', n_jobs=-1, verbose=3, refit=True)
pipe_gridcv.fit(data_train, labels_train)

print(f"[SVM_RFE] Best SVM model with params: {pipe_gridcv.best_params_} and score: {pipe_gridcv.best_score_:.3f}")

# save model
pipe = pipe_gridcv.best_estimator_
save_estimator(pipe, "SVM_RFE_NB.joblib")
print("[SVM_RFE] SVM_RFE model saved")

# predict
score = unbalanced_model_predict(model=pipe, name="SVM_RFE_NB", test_data=data_test, test_labels=labels_test)
print("[SVM_RFE] Balanced accuracy score:", score)

# get BEST features NAMES
get_features_importances_SVMRFE(pipe=pipe, selected_features=selected_features, name='SVM_RFE_NB')

