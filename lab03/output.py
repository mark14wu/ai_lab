import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier

from classifier2 import dnn_model

from xgboost import XGBClassifier

# clf = GradientBoostingClassifier()
clf = XGBClassifier()

data = pd.read_csv("new_train2.csv")

X = data.drop(['pay_more_price','payment'], axis=1).to_numpy()
Y = data['payment'].to_numpy()

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
clf.fit(X, Y)

data = data[data.payment == 1]
data = data.drop(['payment'], axis=1)
X = data.loc[:, data.columns != 'pay_more_price'].to_numpy()
Y = data['pay_more_price'].to_numpy()

scaler2 = preprocessing.StandardScaler().fit(X)
X = scaler2.transform(X)

# clf2 = HuberRegressor().fit(X, Y)


test_data = pd.read_csv("data/tap_fun_test.csv")
X = test_data.drop(['register_time', 'user_id', 'pay_price'], axis=1).to_numpy()
X = scaler.transform(X)
y_class = clf.predict(X)

X = test_data.drop(['register_time', 'user_id', 'pay_price'], axis=1).to_numpy()

# X = scaler2.transform(X)
# y_money = clf2.predict(X)

y_money = dnn_model(X)

print(X.shape)
print(y_money.shape)
print(test_data.shape)

fp = open('result.csv','w')
fp.write('user_id,prediction_pay_price\n')
for i in range(len(test_data)):
    fp.write(str(test_data['user_id'][i])+',')
    if test_data['pay_price'][i] == 0 or test_data['avg_online_minutes'][i] < 15 :
        fp.write('0')
    elif y_class[i] == 0:
        fp.write(str(test_data['pay_price'][i]))
    else:
        fp.write(str(y_money[i][0]+ test_data['pay_price'][i]))
    fp.write('\n')

fp.close()
