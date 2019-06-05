import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

import keras
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.callbacks import TensorBoard

from keras.utils import multi_gpu_model

folds = 5
# clf = SVC(kernel="linear", C=0.25, class_weight={0: 1, 1: 1, 2: 10})
# clf = GradientBoostingClassifier()
#clf = KNeighborsClassifier(3)
# clf = MLPClassifier(alpha=0.5, max_iter=1000, solver='adam', activation='relu')
# clf = GaussianNB(var_smoothing =0.001)
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

    # model = Sequential([
    #     Dense(128, input_dim=107),
    #     Activation('relu'),
    #     Dense(64),
    #     Activation('softmax'),
    #     Dense(48),
    #     Activation('relu'),
    #     Dense(32),
    #     Activation('softmax'),
    #     Dense(48),
    #     Activation('relu'),
    #     Dense(64),
    #     Activation('softmax'),
    #     Dense(128),
    #     Activation('relu'),
    #     Dense(1),
    #     Activation('sigmoid'),
    # ])

    model = Sequential([
        Dense(256, input_dim=107),
        Activation('relu'),
        Dense(128),
        Activation('softmax'),
        Dense(64),
        Activation('softmax'),
        Dense(64),
        Activation('relu'),
        Dense(48),
        Activation('relu'),
        Dense(48),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(1),
        Activation('softmax'),
    ])

    nGPU = 8
    BATCH = 32 * nGPU
    # model = multi_gpu_model(model, gpus=nGPU)
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy'])

    # model.compile(optimizer='adadelta',
    #               loss='mean_squared_error',
    #               metrics=['accuracy'])

    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             batch_size=BATCH,  # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)

    model.fit(
        x=X_train,
        y=Y_train,
        epochs=2000,
        batch_size=BATCH,
        validation_data=(X_test, Y_test),
        callbacks=[tbCallBack]
    )
    print(model.evaluate(X_test, Y_test, batch_size=BATCH))
    exit()

    # clf.fit(X_train, Y_train)
    # scores.append(clf.score(X_test, Y_test))

print(sum(scores) / len(scores))
