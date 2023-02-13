#no balancing
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy, remove_outliers
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from ModelEvaluation.SaveLoad import save_estimator, load_estimator
from ModelEvaluation.Performance import unbalanced_model_predict, select_features_from_model, plot_feature_importance
from sklearn.multiclass import OneVsRestClassifier

#_____________________________________________________________________READ DATASET_____________________________________________________________________#

# Read & Check dataset
data, labels = read_dataset()
check_dataset(data, labels)
data, labels = remove_outliers(data, labels)
data_np, labels_np = dataframe_to_numpy(data, labels)
# Scale the samples
data_sc = scale(data_np)
# Feature Selection
data_np, selected_features = feature_pre_selection(data, data_np)

#_____________________________________________________________________SPLIT DATASET_____________________________________________________________________#


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
rdf_gridcv=GridSearchCV(rdf_model, param_grid=param_grid, cv=4, scoring='balanced_accuracy', error_score='raise', n_jobs=-1, verbose=3, refit=True)
rdf_gridcv.fit(data_train, labels_train)

print(f"[RANDOM FOREST] Best random forest with params: {rdf_gridcv.best_params_} and score: {rdf_gridcv.best_score_:.3f}")
#save model
save_estimator(rdf_gridcv.best_estimator_, "RF_NB.joblib")
print("[RANDOM FOREST] RF_NB model saved")

#predict
score = unbalanced_model_predict(model=rdf_gridcv.best_estimator_, name="RF_NB", test_data=data_test, test_labels=labels_test)
print("[RANDOM_FOREST] Balanced accuracy score:", score)

# plot feature importances for the best model
plot_feature_importance(estimator=rdf_gridcv.best_estimator_, name="RF_NB", selected_features=selected_features)

# select important features based on threshold
imp_features, imp_features_test, feature_names_RFC = select_features_from_model(rdf_gridcv.best_estimator_, 0.0004, True, selected_features, data_train, data_test)
print("[RFC] Found ", len(feature_names_RFC), " important features: ")
print(feature_names_RFC)

#_____________________________________________________________________RETRAIN RANDOMFOREST______________________________________________________________________________#
retrained_rdf = RandomForestClassifier(**rdf_gridcv.best_estimator_.get_params())
retrained_results = retrained_rdf.fit(imp_features, labels_train)

#save model
save_estimator(retrained_rdf, "RF_NB_retrained.joblib")
print("[RANDOM FOREST RETRAINED] RF_NB Re-trained model saved")

#predict
score = unbalanced_model_predict(model=retrained_results, name="RF_NB_retrained", test_data=imp_features_test, test_labels=labels_test)
print("[RANDOM_FOREST RETRAINED] Balanced accuracy score:", score)


#_____________________________________________________________________ONEVSREST-RANDOMFOREST___________________________________________
# select best feature per class
model_rdf = RandomForestClassifier(**rdf_gridcv.best_estimator_.get_params())
ovr = OneVsRestClassifier(estimator=model_rdf, n_jobs=-1)
results = ovr.fit(data_train, labels_train)

#save model
save_estimator(results, "RF_OVR_NB.joblib")
print("[RANDOM FOREST] RF_OVR_NB model saved")

# select important features based on threshold
for i in range(0, 5):
    # plot feature importances for the best model
    plot_feature_importance(estimator=results.estimators_[i], name="RF_OVR_NB", selected_features=selected_features)
    #get important features
    imp_features_train_sin, imp_features_test_sin, feature_names_RFC_sin = select_features_from_model(results.estimators_[i], 0.0004, True, selected_features, data_train, data_test)
    print("[RANDOM FOREST", i, "] Found ", len(feature_names_RFC_sin), " important features")
    print(feature_names_RFC_sin)
