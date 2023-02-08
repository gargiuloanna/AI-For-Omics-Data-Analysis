#no balancing

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from ModelEvaluation.SaveLoad import save_estimator
from ModelEvaluation.Performance import unbalanced_model_predict, select_features_from_model, plot_feature_importance
from sklearn.multiclass import OneVsRestClassifier

directory = "G:/.shortcut-targets-by-id/1H3W_wvBnmy-GZ2KOCF1s1LkjJHPsTlOX/AI-Project/"
#directory = "C:/Users/Luigina/Il mio Drive/AI-Project/"

# Read & Check dataset
data, labels = read_dataset(directory)
check_dataset(data, labels)
data_np, labels_np = dataframe_to_numpy(data, labels)

# Feature Selection
data_np, selected_features = feature_pre_selection(data)

# Scale the samples
data_sc = scale(data_np)

# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_sc, labels_np, test_size=0.30, random_state=12345, stratify=labels_np)
print("[INFO] Finished splitting dataset...")

# _____________________________________________________________________RANDOM FOREST__________________________________________________________________________________#
#Grid Search
print("[RANDOM FOREST] Searching best params with GridSearchCV")

rdf_model=RandomForestClassifier(random_state=12345)
param_grid = {
    'n_estimators': [30, 40, 50, 60, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [0.1, 0.23, 2, 10, 20],
    'min_samples_leaf': [0.1, 0.23, 1, 10, 20],
    'max_features': ['sqrt', 'log2']
}
rdf_gridcv=GridSearchCV(rdf_model, param_grid=param_grid, cv=4, scoring='balanced_accuracy', error_score='raise', n_jobs=-1, verbose=3, refit=False)
rdf_gridcv.fit(data_train, labels_train)

print(f"[RANDOM FOREST] Best random forest with params: {rdf_gridcv.best_params_} and score: {rdf_gridcv.best_score_:.3f}")
# select best feature per class
model_rdf = RandomForestClassifier(**rdf_gridcv.best_estimator_.get_params())
ovr = OneVsRestClassifier(estimator=model_rdf, n_jobs=-1)
results = ovr.fit(data_train, labels_train)

#save model
save_estimator(directory, model_rdf, "RF_NB.joblib")
print("[RANDOM FOREST] RF_NB model saved")

#predict
score = unbalanced_model_predict(model=model_rdf, name="RF_NB", test_data=data_test, test_labels=labels_test, directory=directory)
print("[RANDOM_FOREST] Balanced accuracy score:", score)

# plot feature importances for the best model
plot_feature_importance(estimator = model_rdf, name = "RF_NB", selected_features = selected_features, directory = directory)

# select important features based on threshold
imp_features, imp_features_test, feature_names_RFC = select_features_from_model(rdf_gridcv, 0.0004, True, selected_features, data_train, data_test)
print("[RANDOM FOREST] Found ", len(feature_names_RFC), " important features")

#_________________________________Retraining with selected features________________________#

retrained_rdf = RandomForestClassifier(**model_rdf.get_params())
retrained_ovr = OneVsRestClassifier(estimator=retrained_rdf, n_jobs=-1)
retrained_results = ovr.fit(imp_features, labels_train)

#save model
save_estimator(directory, retrained_rdf, "RF_NB_retrained.joblib")
print("[RANDOM FOREST RETRAINED] RF_NB Re-trained model saved")

#predict
score = unbalanced_model_predict(model=model_rdf, name="RF_NB", test_data=imp_features_test, test_labels=labels_test, directory=directory)
print("[RANDOM_FOREST RETRAINED] Balanced accuracy score:", score)


