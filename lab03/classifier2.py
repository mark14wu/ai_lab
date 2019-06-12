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
from keras.callbacks import EarlyStopping

from keras.utils import multi_gpu_model

from keras.utils.vis_utils import plot_model

import math

# def dnn_model(X, Y):
#     model = Sequential([
#             Dense(64, input_dim=107),
#             Activation('relu'),
#             Dense(32),
#             Activation('relu'),
#             # Dense(64),
#             #Activation('softmax'),
#             #Dense(64),
#             #Activation('relu'),
#            # Dense(64),
#             #Activation('relu'),
#             Dense(64),
#             Activation('relu'),
#             Dense(48),
#             Activation('relu'),
#             Dense(32),
#             Activation('relu'),
#             Dense(32),
#             Activation('relu'),
#             Dense(16),
#             Activation('relu'),
#             Dense(1),
#             Activation('relu'),
#     ])
#     model.compile(
#     loss='mean_squared_error',
#     optimizer='adam',
#     metrics=['accuracy'])
#     model.fit(
#             x=X_train,
#             y=Y_train,
#             epochs=26,
#             batch_size=BATCH,
#             validation_data=(X_test, Y_test),
#             callbacks=[EarlyStopping(monitor='val_loss', patience=4)]
#         )

def dnn_model(input_data):
    folds = 5
    # clf = SVC(kernel="linear", C=0.25, class_weight={0: 1, 1: 1, 2: 10})
    # clf = GradientBoostingClassifier()
    #clf = KNeighborsClassifier(3)
    # clf = MLPClassifier(alpha=0.5, max_iter=1000, solver='adam', activation='relu')
    # clf = GaussianNB(var_smoothing =0.001)
    data = pd.read_csv("new_train2.csv")
    data = data[data.payment == 1]
    data = data.drop(['payment'], axis=1)
    X = data.loc[:, data.columns != 'pay_more_price',].to_numpy()
    Y = data['pay_more_price'].to_numpy()
    group = range(len(Y))
    group = list(map(lambda a: a % folds, group))

    group_kfold = GroupKFold(n_splits=folds)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    input_data = scaler.transform(input_data)
    scores = []

    sum2 = 0

    for train_index, test_index in group_kfold.split(X, Y, group):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # [0.16780971308788054, 0.7648436423790883]
        # model = Sequential([
        #     Dense(128, input_dim=107),
        #     Activation('relu'),
        #     Dense(96),
        #     Activation('relu'),
        #     Dense(64),
        #     Activation('softmax'),
        #     Dense(32),
        #     Activation('softmax'),
        #     Dense(16),
        #     Activation('relu'),
        #     Dense(1),
        #     Activation('relu'),
        # ])

        # [0.16336142523941677, 0.7732470037277317]
        # model = Sequential([
        #     Dense(64, input_dim=107),
        #     Activation('relu'),
        #     Dense(96),
        #     Activation('relu'),
        #     Dense(256),
        #     Activation('softmax'),
        #     Dense(256),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('softmax'),
        #     Dense(64),
        #     Activation('relu'),
        #     Dense(1),
        #     Activation('relu'),
        # ])

        model = Sequential([
                Dense(256, input_dim=105),
                Activation('relu'),
                # Dense(32),
                # Activation('relu'),
                # Dense(64),
                #Activation('softmax'),
                #Dense(64),
                #Activation('relu'),
                # Dense(64),
                # Activation('relu'),
                Dense(512),
                Activation('relu'),
                Dense(256),
                Activation('relu'),
                Dense(128),
                Activation('relu'),
                Dense(32),
                Activation('relu'),
                Dense(1),
                Activation('relu'),
        ])

        nGPU = 20
        # BATCH = 32 * nGPU
        BATCH = 64
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
            epochs=2,
            batch_size=BATCH,
            validation_data=(X_test, Y_test),
            callbacks=[tbCallBack, EarlyStopping(monitor='val_loss', patience=2)]
        )
        y2 = model.predict(X_test)
        sum = 0
        for i in range(len(y2)):
            sum += (y2[i] - Y_test[i])*(y2[i] - Y_test[i])
        sum2+=math.sqrt(sum/len(y2))
        print(sum2)
        break
    
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model.predict(input_data)