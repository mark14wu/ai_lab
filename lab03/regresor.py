import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

folds = 5

clf = HuberRegressor()


data = pd.read_csv("new_train2.csv")
data = data[data.payment == 1]


data = data.drop(['payment'], axis=1)
X = data.loc[:, data.columns != 'pay_more_price'].to_numpy()

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

Y = data['pay_more_price'].to_numpy()
X = SelectFromModel(HuberRegressor()).fit_transform(X,Y)
group = range(len(Y))
group = list(map(lambda a: a % folds, group))

group_kfold = GroupKFold(n_splits=folds)


sum2 =0
for train_index, test_index in group_kfold.split(X, Y, group):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # print(len(X_train))
    # print(len(X_test))
    '''
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('scaler done')
    '''
    clf.fit(X_train, Y_train)
    y2 = clf.predict(X_test)
    sum = 0
    for i in range(len(y2)):
        sum += (y2[i] - Y_test[i])*(y2[i] - Y_test[i])
    sum2+=math.sqrt(sum/len(y2))
    # print(math.sqrt(sum))
print(sum2/5)