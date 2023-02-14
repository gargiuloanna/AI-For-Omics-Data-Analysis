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
data_np, labels_np = dataframe_to_numpy(data, labels)
# Scale the samples
data_sc = scale(data_np)
# Feature Selection
data_np, selected_features = feature_pre_selection(data, data_sc)
# _____________________________________________________________________SPLIT DATASET_____________________________________________________________________#
# Split data
# make sure that the split is always the same,  and that the classes are somewhat balanced between splits
print("[INFO] Splitting dataset...")
data_train, data_test, labels_train, labels_test = train_test_split(data_np, labels_np, test_size=0.30,
                                                                    random_state=12345, stratify=labels_np)
print("[INFO] Finished splitting dataset...")
# _____________________________________________________________________LOAD SVM MODEL_____________________________________________________________________#
# Load SVM_RFE
svm = load_estimator("SVM_RFE_NB.joblib")

# get BEST features NAMES
feature_names_SVM_RFE = get_features_name_RFE(support=svm.named_steps['rfe'].support_, selected_features=selected_features)

# get important features per class
indices = get_importances_sorted(svm.named_steps['svm_model'])
print(indices.shape)
print("[SVM_RFE] Most important features")
#for i in range(5):
    #print('Important Features for class ' + svm.named_steps['svm_model'].classes_[i])
    #print(np.array(feature_names_SVM_RFE)[indices[i]])
set_svm = set(feature_names_SVM_RFE)
# _____________________________________________________________________LOAD RANDOM FOREST_____________________________________________________________________#
# Load Random Forest
rdf = load_estimator("RF_NB.joblib")

# select best feature per class
#this is useful when we only want to see the common features.
imp_features, imp_features_test, feature_names_RFC = select_features_from_model(model=rdf, threshold=0.0004,
                                                                                prefit=True,
                                                                                selected_features=selected_features,
                                                                                train=data_train, test=data_test)


#to see the most important features
f = get_feature_importance(rdf, selected_features, len(feature_names_RFC), "RF_NB")
#print(f)
set_rfc = set(f)
# _____________________________________________________________________COMPARISON_____________________________________________________________________#
intersection = set_svm.intersection(best_set)
intersection2 = set_rfc.intersection(best_set)

print("[INFO] Comparing important features between RF and best set...")
print(len(intersection2), "common features: \n", intersection2)
print("[INFO] Comparing important features between SVM and best set...")
print(len(intersection), "common features: \n", intersection)


'''
# _____________________________________________________________________LOAD RANDOM FOREST-OVR_____________________________________________________________________#
# Load Random Forest
rdf = load_estimator("RF_OVR_NB.joblib")
# _____________________________________________________________________COMPARISON_____________________________________________________________________#
print("[INFO] Comparing important features between RF and SVM for each single tumor...")
for i in range(0, 5):
    imp_features, imp_features_test, feature_names_RFC = select_features_from_model(model=rdf.estimators_[i],
                                                                                    threshold=0.0004, prefit=True,
                                                                                    selected_features=selected_features,
                                                                                    train=data_train, test=data_test)

    f_ovr = get_feature_importance(rdf.estimators_[i], selected_features, len(feature_names_RFC))
    set_rdf = set(feature_names_RFC)
    set_svm = set(np.array(feature_names_SVM_RFE)[indices[i]])
    intersection = set_svm.intersection(set_rdf)
    print('Important Features for class ' + rdf.classes_[i])
    print(len(intersection), "common features: \n", intersection)
'''