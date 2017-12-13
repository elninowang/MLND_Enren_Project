
# P5: 从安然公司邮件中发现欺诈证据


```python
#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")

%matplotlib inline

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
```

    C:\Users\elnin\Anaconda2\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

## 任务0: 浏览数据级，看看数据集大致是什么样


```python
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

count, poi_count = 0,0
all_features_list = set()
for k,v in data_dict.items():
    if v['poi'] == True: poi_count += 1
    count += 1
    for k1 in v.keys():
        if not k1 in all_features_list:
            all_features_list.add(k1)
all_features_list = list(all_features_list)            
print 'count: ',count," poi_count: ",poi_count, "features count:", len(all_features_list)
print "all features coount:", all_features_list
```

    count:  146  poi_count:  18 features count: 21
    all features coount: ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'email_address', 'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'poi', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees']
    

数据集中共有人数 ** 146 ** 人，其中涉嫌欺诈的有 **18** 人, 特征个数是 **21** 个

看看一个典型用户有哪些特征，其值是多少


```python
for k,v in data_dict['METTS MARK'].items():
    print k, " : ", v
```

    salary  :  365788
    to_messages  :  807
    deferral_payments  :  NaN
    total_payments  :  1061827
    exercised_stock_options  :  NaN
    bonus  :  600000
    restricted_stock  :  585062
    shared_receipt_with_poi  :  702
    restricted_stock_deferred  :  NaN
    total_stock_value  :  585062
    expenses  :  94299
    loan_advances  :  NaN
    from_messages  :  29
    other  :  1740
    from_this_person_to_poi  :  1
    poi  :  False
    director_fees  :  NaN
    deferred_income  :  NaN
    long_term_incentive  :  NaN
    email_address  :  mark.metts@enron.com
    from_poi_to_this_person  :  38
    

## 任务1: 选择你需要的特征

选择所有的财务特征, 其中  email_address 是没有任何数据信息，所以可以考虑去掉


```python
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
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
```

    20
    

试试用决策树大致筛选一下对数据的敏感性


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=12)
clf = clf.fit(X, y)
print "clf.feature_importances_ : ", clf.feature_importances_
idx_feature_importances = np.argsort(clf.feature_importances_)[::-1]
for i in range(10):
    idx = idx_feature_importances[i]
    print "importance ", i, " - ", features_list[idx+1], " - ", clf.feature_importances_[idx]
```

    clf.feature_importances_ :  [ 0.          0.          0.16475213  0.          0.17958795  0.          0.0543682
      0.          0.06221847  0.05858439  0.08072355  0.10873641  0.12036185
      0.          0.          0.08721566  0.          0.08345138  0.        ]
    importance  0  -  bonus  -  0.179587954314
    importance  1  -  total_payments  -  0.164752133256
    importance  2  -  restricted_stock  -  0.120361849464
    importance  3  -  long_term_incentive  -  0.108736407949
    importance  4  -  from_poi_to_this_person  -  0.0872156605424
    importance  5  -  from_this_person_to_poi  -  0.0834513822454
    importance  6  -  other  -  0.0807235517364
    importance  7  -  expenses  -  0.0622184676102
    importance  8  -  exercised_stock_options  -  0.0585843889079
    importance  9  -  deferred_income  -  0.0543682039745
    

由此可以看出，和poi关系大的，依次是

- bonus: 奖金
- total_payments: 总收入
- restricted_stock: 受限股票
- expenses: 花费
- long_term_incentive: 长期激励
- from_poi_to_this_person: 从poi邮箱发到此邮箱的邮件数量
- exercised_stock_options: 行权数量
- from_this_person_to_poi: 从此邮箱发到poi邮箱的邮件数量

可以看出比较奇特的是，工资 salary 居然对结果影响不大


```python
new_features_list = ['poi']
for i in range(10):
    idx = idx_feature_importances[i]
    new_features_list.append(features_list[idx+1])
print new_features_list
features_list = new_features_list

```

    ['poi', 'bonus', 'total_payments', 'restricted_stock', 'long_term_incentive', 'from_poi_to_this_person', 'from_this_person_to_poi', 'other', 'expenses', 'exercised_stock_options', 'deferred_income']
    

## 任务2: 删除异常值

由于上面发现的奇怪现象，这里首先检查salary的异常值。

这里先检查 salary 和 bonus里面 有没有NaN的。如果看看，是否存在POI人。


```python
nan_salary_count = 0
nan_bonus_count = 0
poi_count_when_nan_salary_or_nan_bonu = 0
for k,v in data_dict.items():
    if v["salary"] == 'NaN': nan_salary_count +=1
    if v["bonus"] == 'NaN': nan_bonus_count += 1
    if v["salary"] == 'NaN'or v["bonus"] == 'NaN':
        if v["poi"] == True:
            print k," - ", v["salary"], " ", v["bonus"]
print
print "nan_salary_count: ", nan_salary_count, " nan_bonus_count: ", nan_bonus_count
        
```

    YEAGER F SCOTT  -  158403   NaN
    HIRKO JOSEPH  -  NaN   NaN
    
    nan_salary_count:  51  nan_bonus_count:  64
    

发现存在了不少 工资 或者 奖金 NaN, 但是这些里面存在POI人，不能说明是异常


```python
### Task 2: Remove outliers
bonus_idx = 4

max_bonus = np.amax(X[:,bonus_idx])
print "max_bonus ", max_bonus
for k,v in data_dict.items():
    if v["bonus"] == max_bonus:
        print "the max bonus person is ", k
```

    max_bonus  97343619.0
    the max bonus person is  TOTAL
    

TOTAL 是总数，这个是典型的异常值，所以一定要拿掉这个值


```python
data_dict.pop("TOTAL", 0)
y, X = targetFeatureSplit(featureFormat(data_dict, features_list))
X = np.array(X)
print X.shape
```

    (144L, 10L)
    

看看 bonus或者salary 有没有小于0异常值


```python
for k,v in data_dict.items():
    if v["bonus"] == 'NaN' or v["salary"] == 'NaN':
        continue
    if v["bonus"] < 0 or v["salary"] < 0:
        print "the bonus or salary is less than 0 is %s -- %s,%s"%(k,v["bonus"],v["salary"])
print "done"
```

    done
    

看来，没有小于0的，再看看有没有 bonus 和 salary 超级多的


```python
for k,v in data_dict.items():
    if v["salary"] == 'NaN' or v["bonus"] == 'NaN':
        continue
    if v["salary"] > 1000000 and v["bonus"] > 5000000:
        print "the 2 max salary abd bonus is %s -- %s,%s"%(k,v["salary"],v["bonus"])
```

    the 2 max salary abd bonus is LAY KENNETH L -- 1072321,7000000
    the 2 max salary abd bonus is SKILLING JEFFREY K -- 1111258,5600000
    

发现有2个人，这分别是大名鼎鼎的 Kenneth Lay 和 Jeffrey Skilling，很显然，他们不是异常值，他们是正常值

## 任务3: 创建新特征


```python
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
old_features_list = features_list

def isnan(num):
    return num != num
```

增加 从poi邮箱中收取邮件的百分比 和 发送到poi邮箱中的百分比


```python
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
```

增加 奖金和工资的倍数，也就是 `bonus / salary`。按照常识，这个一般奖金按照公司系数发放


```python
for k,v in my_dataset.items():
    #print '=========== ',k
    if float(v['salary']) == 0.: 
        v['bonus_times'] = 0.
    else: 
        v['bonus_times'] = float(v['bonus']) / float(v['salary'])
    if isnan(v['bonus_times']): 
        v['bonus_times'] = 'NaN'
```


```python
### Extract features and labels from dataset for local testing
features_list = old_features_list + ['from_this_to_poi_ratio', 'from_poi_to_this_ratio', 'bonus_times']

data = featureFormat(my_dataset, features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
X = np.array(X)
print "features ", X.shape
print "lables ", len(y)
```

    features  (144L, 13L)
    lables  144
    

特征增加了，再次用决策树看看，重要相关特征有哪些


```python
clf = tree.DecisionTreeClassifier(random_state=12)
clf = clf.fit(X, y)
print "clf.feature_importances_ : ", clf.feature_importances_
idx_feature_importances = np.argsort(clf.feature_importances_)[::-1]
for i in range(10):
    idx = idx_feature_importances[i]
    print "importance ", i, " - ", features_list[idx+1], " - ", clf.feature_importances_[idx]
```

    clf.feature_importances_ :  [ 0.04232804  0.          0.02817127  0.04232804  0.          0.          0.
      0.27209809  0.27220666  0.08300395  0.25986395  0.          0.        ]
    importance  0  -  exercised_stock_options  -  0.272206660442
    importance  1  -  expenses  -  0.272098088247
    importance  2  -  from_this_to_poi_ratio  -  0.259863945578
    importance  3  -  deferred_income  -  0.0830039525692
    importance  4  -  long_term_incentive  -  0.042328042328
    importance  5  -  bonus  -  0.042328042328
    importance  6  -  restricted_stock  -  0.0281712685074
    importance  7  -  bonus_times  -  0.0
    importance  8  -  from_poi_to_this_ratio  -  0.0
    importance  9  -  other  -  0.0
    

根据上面的内容，我把特征的范围再缩小一下，只用决策树有用的特征


```python
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
```

    ['poi', 'exercised_stock_options', 'expenses', 'from_this_to_poi_ratio', 'deferred_income', 'long_term_incentive', 'bonus', 'restricted_stock']
    features  (141L, 7L)
    lables  141
    

## 任务4: 尝试各种各样的分类器模型

初始化一些公有的东西,包括PCA转换，MinMaxScalar 还有 StratifiedShuffleSplit

**为了节约性能** StratifiedShuffleSplit 我只设置了100次的迭代


```python
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
```

    done
    

后面开始尝试各种各样的算法， 对于每一种算法，我分别尝试 PAC不转化 和 PAC转化的

由于项目要求的 precision 和 recall，我采用f1 score作为GridSearchCV的评估标准，原因是

f1 = 2 * precision * recall / (precision + recall) 这样f1 score同时兼顾了 precision 和 recall 两个指标

### 尝试朴素贝叶斯算法


```python
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
```

    before pca f1:  0.413857142857
    after pca f1:  0.377666666667
    

### 尝试决策树算法


```python
# DissionTree
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=12)
parameters = {'max_depth': range(3,10)}
tree_clf_grid = GridSearchCV(tree_clf, parameters, cv=cv, scoring='f1')
tree_clf_grid.fit(X, y)
print "before pca f1: ", tree_clf_grid.best_score_

pca_tree_clf = Pipeline([('pca', pca), ('svc', tree_clf)])
parameters = {'pca__n_components': range(1,5),
              'svc__max_depth': range(3,10)}
pca_tree_clf_grid = GridSearchCV(pca_tree_clf, parameters, cv=cv, scoring='f1')
pca_tree_clf_grid.fit(X, y)
print "after pca f1: ", pca_tree_clf_grid.best_score_
```

    before pca f1:  0.193714285714
    after pca f1:  0.38946031746
    

### 尝试KNN算法


```python
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
parameters = {'n_neighbors': [3,5,7,10,15],
              'weights': ['uniform', 'distance']}
knn_clf_grid = GridSearchCV(knn_clf, parameters, cv=cv, scoring='f1')
knn_clf_grid.fit(X, y)
print "before pca f1: ", knn_clf_grid.best_score_

pca_knn_clf = Pipeline([('pca', pca), ('clf', knn_clf)])
parameters = {'pca__n_components': range(1,5),
              'clf__n_neighbors': [3,5,7,10,15],
              'clf__weights': ['uniform', 'distance']}
pca_knn_clf_grid = GridSearchCV(pca_knn_clf, parameters, cv=cv, scoring='f1')
pca_knn_clf_grid.fit(X, y)
print "after pca f1: ", pca_knn_clf_grid.best_score_
```

    before pca f1:  0.267
    after pca f1:  0.314333333333
    

### 尝试SVM算法

** 很奇怪，svm 一直计算，远出不来结果，只能暂时屏蔽掉** 原因不明


```python
# # svm
# from sklearn.svm import SVC
# svm_clf = SVC(random_state=42)
# parameters = {'kernel': ('rbf','linear'), 
#               'C': [1,10]}
# svm_clf_grid = GridSearchCV(svm_clf, parameters, cv=cv, scoring='f1')
# svm_clf_grid.fit(X, y)
# print "before pca f1: ", svm_clf_grid.best_score_

# pca_svm_clf = Pipeline([('pca', pca), ('svc', svm_clf)])
# parameters = {'pca__n_components': [3,5,7,9],
#               'svc__kernel': ['rbf','linear'], 
#               'svc__C': [1,10]
#              }
# pca_svm_clf_grid = GridSearchCV(pca_svm_clf, parameters, cv=cv, scoring='f1')
# pca_svm_clf_grid.fit(X, y)
# print "after pca f1: ", pca_svm_clf_grid.best_score_
```

### 尝试一些集成学习算法，AdaBoost


```python
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(random_state=42)
parameters = {'n_estimators': [25,50,75,100],
              'learning_rate': [0.01, 0.1, 1]}
adaboost_clf_grid = GridSearchCV(adaboost_clf, parameters, cv=cv, scoring='f1')
adaboost_clf_grid.fit(X, y)
print "before pca f1: ", adaboost_clf_grid.best_score_

pca_adaboost_clf = Pipeline([('pca', pca), ('svc', adaboost_clf)])
parameters = {'pca__n_components': range(1,5),
              'svc__n_estimators': [25,50,75,100],
              'svc__learning_rate': [0.01, 0.1, 1]}
pca_adaboost_clf_grid = GridSearchCV(pca_adaboost_clf, parameters, cv=cv, scoring='f1')
pca_adaboost_clf_grid.fit(X, y)
print "after pca f1: ", pca_adaboost_clf_grid.best_score_
```

    before pca f1:  0.434714285714
    after pca f1:  0.361714285714
    

### 尝试一些集成学习算法，RandomForest


```python
# RandomForest
from sklearn.ensemble import RandomForestClassifier
forest_clf = AdaBoostClassifier(random_state=42)
parameters = {'n_estimators': [5,10,15,20]}
forest_clf_grid = GridSearchCV(forest_clf, parameters, cv=cv, scoring='f1')
forest_clf_grid.fit(X, y)
print "before pca f1: ", forest_clf_grid.best_score_

pca_forest_clf = Pipeline([('pca', pca), ('svc', forest_clf)])
parameters = {'pca__n_components': range(1,5),
              'svc__n_estimators': [5,10,15,20]}
pca_forest_clf_grid = GridSearchCV(pca_forest_clf, parameters, cv=cv, scoring='f1')
pca_forest_clf_grid.fit(X, y)
print "after pca f1: ", pca_forest_clf_grid.best_score_
```

    before pca f1:  0.386095238095
    after pca f1:  0.343380952381
    

## 任务5：用分类器做预测

根据之上尝试过的算法来看，大致看来，使用朴素贝叶斯且不经过PCA转换的f1 score效果最好。

所以我将最终算法设定为朴素贝叶斯


```python

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = naive_clf_grid.best_estimator_
print "final clf: ", clf
```

    final clf:  GaussianNB(priors=None)
    

后面拆分训练集和测试集数据 来简单验证一下，看看相关分数是多少


```python
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
```

    accuracy:  0.883720930233  precision: 0.5  recall: 0.6  f1: 0.545454545455
    

可见其，测试下载，其 precision 和 recall 分别能做到 0.5 和 0.6

另外，运行 tester.py 其结果如下：

```
GaussianNB(priors=None)
	Accuracy: 0.85767	Precision: 0.45841	Recall: 0.37200	F1: 0.41071	F2: 0.38657
	Total predictions: 15000	True positives:  744	False positives:  879	False negatives: 1256	True negatives: 12121
```

## 任务6: 把分类器，数据集，和特征列表存在本地，分别是

- my_classifier.pkl 是最终选择的分类器
- my_dataset.pkl 是数据集
- my_feature_list.pkl 是选择的特征


```python
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print "done"
```

    done
    


```python
print "done"
```

    done
    


```python

```
