#!/usr/bin/python

import sys
import pickle

import tester
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','total_payments','bonus','total_stock_value','expenses',
                 'deferred_income','expenses','exercised_stock_options','long_term_incentive','shared_receipt_with_poi',
                 'restricted_stock']

# removed features: restricted_stock_deferred, loan_advances, director_fees, deferral_payments

### Load the dictionary containing the dataset
''' #### MAY NEED TO UNCOMMENT IF ON WINDOWS - THIS WAS A MAJOR ISSUE ####
content = ''
outsize = 0
with open('final_project_dataset.pkl', 'rb') as infile:
    content = infile.read()
with open('final_project_dataset_unix.pkl', 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))
'''
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
print("Num data points:", len(data_dict))
print("Num features:", len(data_dict['TOTAL']))

### Task 2: Remove outliers
del data_dict['TOTAL']
names_outliers = ['LAY KENNETH L', 'LAY KENNETH L', 'LAY KENNETH L', 'LAY KENNETH L', 'SKILLING JEFFREY K',
                  'WHITE JR THOMAS E', 'FREVERT MARK A', 'FREVERT MARK A', 'FREVERT MARK A', 'FREVERT MARK A',
                  'PICKERING MARK R', 'MARTIN AMANDA K', 'MARTIN AMANDA K', 'LAVORATO JOHN J', 'MARTIN AMANDA K',
                  'LAVORATO JOHN J', 'LAVORATO JOHN J', 'MCCLELLAN GEORGE', 'BHATNAGAR SANJAY', 'HORTON STANLEY C',
                  'BAXTER JOHN C', 'BELDEN TIMOTHY N', 'HIRKO JOSEPH', 'FREVERT MARK A', 'SKILLING JEFFREY K',
                  'LAY KENNETH L', 'LAY KENNETH L', 'HUMPHREY GENE E', 'PAI LOU L', 'BHATNAGAR SANJAY',
                  'BHATNAGAR SANJAY', 'LAVORATO JOHN J']
features_outliers = ['total_payments', 'salary', 'restricted_stock','expenses', 'total_payments', 'restricted_stock',
                     'salary', 'loan_advances', 'bonus', 'total_payments','loan_advances', 'deferral_payments',
                     'total_payments', 'long_term_incentive', 'long_term_incentive','bonus', 'expenses',
                     'total_payments', 'expenses', 'restricted_stock', 'restricted_stock', 'restricted_stock',
                     'shared_receipt_with_poi', 'exercised_stock_options', 'restricted_stock', 'restricted_stock',
                     'long_term_incentive', 'total_stock_value', 'percent_person_to_poi', 'restricted_stock',
                     'total_payments', 'total_payments']

numpoi = 0
for name in data_dict:
    if data_dict[name]['poi'] == 1:
        numpoi += 1
print("Num POI:", numpoi)
### Task 3: Create new feature(s)
''' #### 2 NEW FEATURES -> PERCENTPOITOPERSON, PERCENTPERSONTOPOI #### '''
for name in data_dict:
    L = data_dict[name]
    if L['to_messages'] != 'NaN' and L['from_poi_to_this_person'] != 'NaN':
        L['percent_poi_to_person'] = L['from_poi_to_this_person'] / L['to_messages']
    else:
        L['percent_poi_to_person'] = 'NaN'
    if L['from_messages'] != 'NaN' and L['from_this_person_to_poi'] != 'NaN':
        L['percent_person_to_poi'] = L['from_this_person_to_poi'] / L['from_messages']
    else:
        L['percent_person_to_poi'] = 'NaN'

''' #### Removing Outliers #### '''
for i in range(len(names_outliers)):
    data_dict[names_outliers[i]][features_outliers[i]] = 'NaN'

''' #### Rescaling Data #### '''
max_features = [-1e9] * len(features_list)
min_features = [1e9] * len(features_list)
for name in data_dict:
    for i in range(len(features_list)):
        if data_dict[name][features_list[i]] != 'NaN':
            max_features[i] = max(max_features[i], data_dict[name][features_list[i]])
            min_features[i] = min(min_features[i], data_dict[name][features_list[i]])
for name in data_dict:
    for i in range(len(features_list)):
        if data_dict[name][features_list[i]] != 'NaN':
            data_dict[name][features_list[i]] = (data_dict[name][features_list[i]] - min_features[i]) /(max_features[i] - min_features[i])

features_list.append('percent_poi_to_person')
features_list.append('percent_person_to_poi')

''' #### Graph Relationships Between Features ####
import matplotlib.pyplot as plt
for i in range(1, len(features_list)):
    for j in range(1, len(features_list)):
        if i != j and i < len(features_list) and j < len(features_list):
            print(features_list[i], features_list[j])
            f1 = features_list[i]
            f2 = features_list[j]
            npoif1 = []
            npoif2 = []
            poif1 = []
            poif2 = []
            npoilabels = []
            poilabels = []
            for name in data_dict:
                if data_dict[name][f1] != 'NaN' and data_dict[name][f2] != 'NaN':
                    if data_dict[name]['poi'] == 0:
                        npoif1.append(data_dict[name][f1])
                        npoif2.append(data_dict[name][f2])
                        npoilabels.append(name)
                    else:
                        poif1.append(data_dict[name][f1])
                        poif2.append(data_dict[name][f2])
                        poilabels.append(name)
            print(npoif1, npoif2)
            plt.scatter(npoif1, npoif2, color='blue')
            plt.scatter(poif1, poif2, color='red')
            for i, txt in enumerate(npoilabels):
                plt.annotate(txt, (npoif1[i], npoif2[i]))
            for i, txt in enumerate(poilabels):
                plt.annotate(txt, (poif1[i], poif2[i]))
            plt.suptitle(f1 + " vs " + f2)
            plt.xlabel(f1)
            plt.ylabel(f2)
            plt.show()
'''

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
from sklearn.feature_selection import SelectPercentile
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score, f1_score
clf = Pipeline(steps=[('pca', PCA()), ('fselect', SelectPercentile()), ('clf', GaussianNB())])
param_grid = {
    'pca__n_components': [5, 7, 9, 11, 13],
    'fselect__percentile': [33, 50, 75, 100],
}
''' #### Random Forest ####
clf = Pipeline(steps=[('pca', PCA()), ('fselect', SelectPercentile()), ('clf', RandomForestClassifier())])
param_grid = {
    'pca__n_components': [5, 7, 9, 11, 13],
    'fselect__percentile': [33, 50, 75, 100],
    'clf__n_estimators': [1, 3, 5, 10, 15, 20, 50, 100]
}
'''
search = GridSearchCV(clf, param_grid, n_jobs=2)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
search.fit(features_train, labels_train)
params = search.best_params_
print("PCA Components:", params['pca__n_components'])
print("Percent Components Used: ", params['fselect__percentile'], "%", sep='')
''' #### Random Forest ####
clf = Pipeline(steps=[('pca', PCA(params['pca__n_components'])), ('fselect', SelectPercentile(percentile=params[
    'fselect__percentile'])), ('clf', RandomForestClassifier(n_estimators=params['clf__n_estimators']))])
'''
clf = Pipeline(steps=[('pca', PCA(params['pca__n_components'])), ('fselect', SelectPercentile(percentile=params[
    'fselect__percentile'])), ('clf', GaussianNB())])
clf.fit(features_train, labels_train)
print("Feature scores for my algorithm:", [i for i in reversed(sorted(clf.steps[1][1].scores_))][:3])
pred = clf.predict(features_test)
print("Accuracy score:", accuracy_score(pred, labels_test))
print("F1 score:", f1_score(pred, labels_test))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)