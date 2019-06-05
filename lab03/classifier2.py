import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

folds = 5
# clf = SVC(kernel="linear", C=0.25, class_weight={0: 1, 1: 1, 2: 10})
# clf = GradientBoostingClassifier()
#clf = KNeighborsClassifier(3)
# clf = MLPClassifier(alpha=0.5, max_iter=1000, solver='adam', activation='relu')
clf = GaussianNB(var_smoothing =0.001)
data = pd.read_csv("new_train2.csv")
X = data.loc[:, data.columns != 'payment'].to_numpy()
Y = data['payment'].to_numpy()
group = range(len(Y))
group = list(map(lambda a: a % folds, group))

group_kfold = GroupKFold(n_splits=folds)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
scores = []
for train_index, test_index in group_kfold.split(X, Y, group):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print(len(X_train))
    print(len(X_test))
    '''
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('scaler done')
    '''
    clf.fit(X_train, Y_train)
    scores.append(clf.score(X_test, Y_test))

print(sum(scores) / len(scores))
