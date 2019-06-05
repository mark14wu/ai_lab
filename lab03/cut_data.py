import pandas as pd
train_data = pd.read_csv("data/tap_fun_train.csv")
dir_name = 'train'
'''
def show_num(data):
    temp = pd.cut(pd.Series(data['prediction_pay_price']),[0,10,100,1000,100000],right=True)
    print('0:'+str(len(data[data.prediction_pay_price == 0]))+'\n')
    print(pd.value_counts(temp))


temp1 = train_data[(train_data.pay_price < 1) & (train_data.avg_online_minutes<10)]
show_num(temp1)
show_num(train_data)
'''
new_train_set = train_data[(train_data.pay_price > 0 )]
new_train_set = new_train_set.drop(['register_time', 'user_id'], axis=1)
new_train_set['pay_more_price'] = (new_train_set['prediction_pay_price'] - new_train_set['pay_price'])

print(len(new_train_set))

temp1 = new_train_set[new_train_set.pay_more_price == 0].sample(frac=1)
temp1['payment'] = 0
print(len(temp1))

temp2 = new_train_set[(new_train_set.pay_more_price < 100) & (new_train_set.pay_more_price > 0)].sample(frac=1)
temp2['payment'] = 1
print(len(temp2))

temp3 = new_train_set[(new_train_set.pay_more_price >= 100)].sample(frac=1)
temp3['payment'] = 2
print(len(temp3))
temp4 = pd.concat([temp1, temp2, temp3], axis=0)
print(len(temp4))
temp4 = temp4.drop(['prediction_pay_price', 'pay_more_price'], axis=1)
temp4.to_csv('new_train.csv')
# 其实以上部分只要一个map就好了