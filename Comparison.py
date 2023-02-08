#no balancing

from ModelEvaluation.SaveLoad import load_estimator
from ModelEvaluation.Performance import unbalanced_model_predict, select_features_from_model, plot_feature_importance
from sklearn.model_selection import train_test_split
from DatasetPrep.DatasetPreparation import read_dataset, check_dataset, dataframe_to_numpy
from DatasetPrep.VariablePreSelection import feature_pre_selection
from DatasetPrep.Scaling import scale
from sklearn.ensemble import RandomForestClassifier
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

#Load SVM_RFE
svm = load_estimator(directory, "SVM_RFE_NB.joblib")
#get BEST features NAMES
mask = svm.named_steps['rfe'].support_
feature_names_SVM_RFE = []
for i in range(len(mask)):
  if mask[i] == True:
    feature_names_SVM_RFE.append(selected_features[i])

#get important features per class
c = svm.named_steps['svm_model'].coef_
print("[SVM_RFE]")
for i in range(5):
  print('Important Features for class ' + svm.named_steps['svm_model'].classes_[i])
  print(feature_names_SVM_RFE[c[i].argmax()])
  print(feature_names_SVM_RFE[c[i].argmin()])
set_svm = set(feature_names_SVM_RFE)

#Load Random Forest
rdf = load_estimator(directory, "RF_NB.joblib")
# select best feature per class
model_rdf = RandomForestClassifier(**rdf.get_params())
ovr = OneVsRestClassifier(estimator=model_rdf, n_jobs=-1)
rdf_fit = ovr.fit(data_train, labels_train)
print(rdf_fit.estimators_)

for i in range(0, 5):
  imp_features, imp_features_test, feature_names_RFC = select_features_from_model(model=rdf_fit.estimators_[i], threshold=0.0004, prefit=True, selected_features=selected_features, train = data_train, test = data_test)
  #print("[RANDOM FOREST" , i , "] Found ", len(feature_names_RFC), " important features:\n", feature_names_RFC)
  set_rfc = set(feature_names_RFC)
  print(feature_names_SVM_RFE[c[i].argmax()])
  print(feature_names_SVM_RFE[c[i].argmin()])
  set_svm = set([feature_names_SVM_RFE[c[i].argmax()],feature_names_SVM_RFE[c[i].argmin()]])
  intersection = set_svm.intersection(set_rfc)
  print("length of intersection,  ", len(intersection))
  print("Common features: \n", intersection)










'''
# select important features based on threshold
imp_features, imp_features_test, feature_names_RFC = select_features_from_model(model=rdf, threshold=0.0004, prefit=True, selected_features=selected_features, train = data_train, test = data_test)
print("[RANDOM FOREST] Found ", len(feature_names_RFC), " important features")
'''




'''
print("number of features of SVM, " , len(set_svm))
print("number of features of RFC, " , len(set_rfc))
intersection = set_svm.intersection(set_rfc)
print("length of intersection,  ", len(intersection))
print("Common features: \n", intersection)
'''