#balancing
from imblearn.under_sampling import NearMiss
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, remove_outliers
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from ModelEvaluation.SaveLoad import save_estimator
from ModelEvaluation.Performance import unbalanced_model_predict, select_features_from_model, plot_feature_importance, get_features_name

#_____________________________________________________________________READ DATASET_____________________________________________________________________#
# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
data, labels = remove_outliers(data, labels)
nearmiss = NearMiss(version=3, n_jobs=-1)
data_resampled_np, labels_resampled_np = nearmiss.fit_resample(data, labels)
print("Total number of samples after smote: ", len(data_resampled_np), ". Total number of labels ", len(labels_resampled_np))
# Scale the samples
data_sc = scale(data_resampled_np)
# Feature Selection
data_np, selected_features = feature_pre_selection(data, data_resampled_np)

#_____________________________________________________________________SPLIT DATASET_____________________________________________________________________#
# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_np, labels_resampled_np, test_size=0.30, random_state=12345, stratify=labels_resampled_np)
print("[INFO] Finished splitting dataset...")

#_____________________________________________________________________SVM_RFE__________________________________________________________________________________#
#grid search with RFE
print("[SVM_RFE] Searching best params with GridSearchCV")

svm_model = LinearSVC(max_iter=10000, random_state=12345)
rfe = RFECV(svm_model, step=10000, verbose=2)
pipe = Pipeline([('rfe', rfe), ('svm_model', svm_model)])
param_grid = {'svm_model__C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
              'svm_model__loss': ['hinge', 'squared_hinge']}

pipe_gridcv = GridSearchCV(pipe, param_grid=param_grid, cv=4, scoring='balanced_accuracy', error_score='raise', n_jobs=-1, verbose=3, refit=True)
pipe_gridcv.fit(data_train, labels_train)

print(f"[SVM_RFE] Best SVM model with params: {pipe_gridcv.best_params_} and score: {pipe_gridcv.best_score_:.3f}")

#save model
pipe = pipe_gridcv.best_estimator_
save_estimator(pipe, "SVM_RFE_NB.joblib")
print("[SVM_RFE] SVM_RFE model saved")

#predict
score = unbalanced_model_predict(model=pipe, name="SVM_RFE_NB", test_data=data_test, test_labels=labels_test)
print("[SVM_RFE] Balanced accuracy score:", score)

# plot feature importances for the best model
plot_feature_importance(estimator=pipe, name="SVM_RFE_NB", selected_features=selected_features)

# select important features based on threshold
imp_features, imp_features_test, feature_names_SVM = select_features_from_model(pipe, 0.0004, True, selected_features, data_train, data_test)
print("[SVM_RFE] Found ", len(feature_names_SVM), " important features")

#get BEST features NAMES
feature_names_SVM_RFE = get_features_name(support=pipe.named_steps['rfe'].support_, selected_features=selected_features)

#get important features per class
c = pipe.named_steps['svm_model'].coef_
print("[SVM_RFE]")
for i in range(5):
  print('Important Features for class ' + pipe.named_steps['svm_model'].classes_[i])
  print(feature_names_SVM_RFE[c[i].argmax()])
  print(feature_names_SVM_RFE[c[i].argmin()])