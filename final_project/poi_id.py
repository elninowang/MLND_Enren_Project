#!/usr/bin/python

# all code is in poi_id.ipynb

import sys
import pickle
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

features_list = ['poi']
# finance feature
features_list += ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                  'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                  'director_fees']
# email feature
features_list +=  ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

print len(features_list)

### Load the dictionary containing the dataset
y, X = targetFeatureSplit(featureFormat(data_dict, features_list))
X = np.array(X)

from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=12)
clf = clf.fit(X, y)
print "clf.feature_importances_ : ", clf.feature_importances_
idx_feature_importances = np.argsort(clf.feature_importances_)[::-1]
for i in range(10):
    idx = idx_feature_importances[i]
    print "importance ", i, " - ", features_list[idx+1], " - ", clf.feature_importances_[idx]

new_features_list = ['poi']
for i in range(10):
    idx = idx_feature_importances[i]
    new_features_list.append(features_list[idx+1])
print new_features_list
features_list = new_features_list

### Task 2: Remove outliers
bonus_idx = 4

max_bonus = np.amax(X[:,bonus_idx])
print "max_bonus ", max_bonus
for k,v in data_dict.items():
    if v["bonus"] == max_bonus:
        print "the max bonus person is ", k

data_dict.pop("TOTAL", 0)
y, X = targetFeatureSplit(featureFormat(data_dict, features_list))
X = np.array(X)
print X.shape

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
old_features_list = features_list

def isnan(num):
    return num != num

for k,v in my_dataset.items():
    if float(v['to_messages']) == 0.:
        v['from_poi_to_this_ratio'] = 0.
    else:
        v['from_poi_to_this_ratio'] = float(v['from_poi_to_this_person']) / float(v['to_messages'])
    if float(v['from_messages']) == 0.:
        v['from_this_to_poi_ratio'] = 0.
    else:
        v['from_this_to_poi_ratio'] =  float(v['from_this_person_to_poi']) / float(v['from_messages'])
    if isnan(v['from_poi_to_this_ratio']): v['from_poi_to_this_ratio'] = 'NaN'
    if isnan(v['from_this_to_poi_ratio']): v['from_this_to_poi_ratio'] = 'NaN'

for k,v in my_dataset.items():
    #print '=========== ',k
    if float(v['salary']) == 0.:
        v['bonus_times'] = 0.
    else:
        v['bonus_times'] = float(v['bonus']) / float(v['salary'])
    if isnan(v['bonus_times']):
        v['bonus_times'] = 'NaN'

### Extract features and labels from dataset for local testing
features_list = old_features_list + ['from_this_to_poi_ratio', 'from_poi_to_this_ratio', 'bonus_times']

data = featureFormat(my_dataset, features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
X = np.array(X)
print "features ", X.shape
print "lables ", len(y)

clf = tree.DecisionTreeClassifier(random_state=12)
clf = clf.fit(X, y)
print "clf.feature_importances_ : ", clf.feature_importances_
idx_feature_importances = np.argsort(clf.feature_importances_)[::-1]
for i in range(10):
    idx = idx_feature_importances[i]
    print "importance ", i, " - ", features_list[idx+1], " - ", clf.feature_importances_[idx]

new_features_list = ['poi']
for i in range(7):
    idx = idx_feature_importances[i]
    new_features_list.append(features_list[idx+1])
features_list = new_features_list
print new_features_list

data = featureFormat(my_dataset, features_list, remove_NaN=True)
y, X = targetFeatureSplit(data)
X = np.array(X)
print "features ", X.shape
print "lables ", len(y)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
pca = PCA(svd_solver='randomized', random_state=42)
cv = StratifiedShuffleSplit(y, 100, random_state=42)
print "done"

# GaussianNB
from sklearn.naive_bayes import GaussianNB
naive_clf = GaussianNB()
parameters = {}
naive_clf_grid = GridSearchCV(naive_clf, parameters, cv=cv, scoring='f1')
naive_clf_grid.fit(X, y)
print "before pca f1: ", naive_clf_grid.best_score_

pca_naive_clf = Pipeline([('pca', pca),('svc', naive_clf)])
parameters = {'pca__n_components': range(1,5)}
pca_naive_clf_grid = GridSearchCV(pca_naive_clf, parameters, cv=cv, scoring='f1')
pca_naive_clf_grid.fit(X, y)
print "after pca f1: ", pca_naive_clf_grid.best_score_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = naive_clf_grid.best_estimator_
print "final clf: ", clf

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print "accuracy: ",accuracy, " precision:",precision, " recall:", recall, " f1:", f1

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print "done"