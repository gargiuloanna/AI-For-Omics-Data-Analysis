from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
import numpy as np
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy, remove_outliers
from DatasetPrep.Scaling import scale
from DatasetPrep.VariablePreSelection import feature_pre_selection
from ModelEvaluation.Performance import select_features_from_model, get_features_name_RFE, get_feature_importance, get_importances_sorted
from ModelEvaluation.SaveLoad import load_estimator

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
data_np, selected_features = feature_pre_selection(data, data_sc)
# _____________________________________________________________________SPLIT DATASET_____________________________________________________________________#
# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_np, labels_resampled_np, test_size=0.30,
                                                                    random_state=12345, stratify=labels_resampled_np)
print("[INFO] Finished splitting dataset...")
# _____________________________________________________________________LOAD SVM MODEL_____________________________________________________________________#
# Load SVM_RFE
svm = load_estimator("SVM_RFE_NEARMISS.joblib")

# get BEST features NAMES
feature_names_SVM_RFE = get_features_name_RFE(support=svm.named_steps['rfe'].support_, selected_features=selected_features)

# get important features per class
indices = get_importances_sorted(svm.named_steps['svm_model'])
print(indices.shape)
print("[SVM_RFE] Most important features")
for i in range(5):
    print('Important Features for class ' + svm.named_steps['svm_model'].classes_[i])
    print(np.array(feature_names_SVM_RFE)[indices[i]])
set_svm = set(feature_names_SVM_RFE)
# _____________________________________________________________________LOAD RANDOM FOREST_____________________________________________________________________#
# Load Random Forest
rdf = load_estimator("RF_NEARMISS.joblib")

# select best feature per class
#this is useful when we only want to see the common features.
imp_features, imp_features_test, feature_names_RFC = select_features_from_model(model=rdf, threshold=0.0004,
                                                                                prefit=True,
                                                                                selected_features=selected_features,
                                                                                train=data_train, test=data_test)

#to see the most important features
f = get_feature_importance(rdf, selected_features, len(feature_names_RFC))
print(f)
set_rfc = set(f)
# _____________________________________________________________________COMPARISON_____________________________________________________________________#
intersection = set_svm.intersection(set_rfc)
print("[INFO] Comparing important features between RF and SVM for all the five tumors...")
print(len(intersection), "common features: \n", intersection)



# _____________________________________________________________________LOAD RANDOM FOREST-OVR_____________________________________________________________________#
# Load Random Forest
rdf = load_estimator("RF_OVR_NEARMISS.joblib")
# _____________________________________________________________________COMPARISON_____________________________________________________________________#
print("[INFO] Comparing important features between RF and SVM for each single tumor...")
for i in range(0, 5):
    imp_features, imp_features_test, feature_names_RFC = select_features_from_model(model=rdf.estimators_[i],
                                                                                    threshold=0.0004, prefit=True,
                                                                                    selected_features=selected_features,
                                                                                    train=data_train, test=data_test)

    f_ovr = get_feature_importance(rdf.estimators_[i], selected_features, len(feature_names_RFC))
    set_rdf = set(f_ovr)
    set_svm = set(np.array(feature_names_SVM_RFE)[indices[i]])
    intersection = set_svm.intersection(set_rdf)
    print('Important Features for class ' + rdf.classes_[i])
    print(len(intersection), "common features: \n", intersection)
