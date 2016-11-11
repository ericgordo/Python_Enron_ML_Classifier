#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

temp_features = ['poi','to_messages', 
                 'deferral_payments','combined_emails', 'expenses',
                 'long_term_incentive', 'from_poi_to_this_person',
                 'deferred_income','restricted_stock_deferred', 
                 'shared_receipt_with_poi', 'loan_advances', 
                 'from_messages', 'other','director_fees', 
                 'bonus', 'total_stock_value', 
                 'from_this_person_to_poi', 'restricted_stock',
                 'salary', 'total_payments', 'exercised_stock_options']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Create new feature(s)
e_mail_labels=['from_messages','from_poi_to_this_person','to_messages','from_this_person_to_poi']
for person in data_dict.keys():
   for label in e_mail_labels:
       if data_dict[person][label]== 'NaN':
           data_dict[person]['combined_emails']= 0
       else:
           data_dict[person]['combined_emails'] = \
           data_dict[person]['from_messages'] + \
           data_dict[person]['from_poi_to_this_person'] + \
           data_dict[person]['to_messages'] + \
           data_dict[person]['from_this_person_to_poi']
     
### Task 2: Remove outliers
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
data_dict.pop('TOTAL', None)
data_dict.pop('LOCKHART EUGENE E', None)

### Store to my_dataset for easy export below.
my_dataset = data_dict

#Create Temperary Data   
data = featureFormat(my_dataset, temp_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 3: Select what features you'll use.
features_list=['poi']
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
feature_counts={}

kf=StratifiedShuffleSplit(labels, n_iter=5, random_state=50)
for train_indices, test_indices in kf:
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train= [labels[ii] for ii in train_indices]
    labels_test= [labels[ii] for ii in test_indices]
    
    kbest=SelectKBest(k=14)
    scaler=MinMaxScaler()
    features_scaled = scaler.fit_transform(features_train, labels_train )
    kbest.fit(features_scaled, labels_train )
    features_to_use=kbest.get_support()
    
    for i, item in enumerate(features_to_use):
        if item:
            feature=temp_features[i + 1]
            if feature in feature_counts:
                feature_counts[feature]+=1
            else:
                feature_counts[feature]=1

for feature in feature_counts:
    if feature_counts[feature] > 1:
        features_list.append(feature)

print "\n Features List Set To: \n",len(features_list), features_list

# ReCreate Data With New Feature List
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
#Look At Jupyter Notebook for Tries of Multiple Classifiers

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

clf=Pipeline(steps=[('PCA', PCA(copy=True, n_components=8, whiten=False)),
                    ('Classifier', DecisionTreeClassifier(
                       class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=30, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'))])
            
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. 
dump_classifier_and_data(clf, my_dataset, features_list)
