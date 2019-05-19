import warnings

warnings.filterwarnings("ignore")

import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from preprocessing import getDataDEAP, getDataMAHNOB_HCI, getNBDataMAHNOB_HCI, \
    getNBDataDEAP, getEmotionDataMAHNOB_HCI



def train_MLP(x_train, y_train, x_test, y_test):
    input_size = len(x_train[0])
    output_size = len(y_train[0])
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_size))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    earlystop = [EarlyStopping(patience=4)]

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=1000,
              batch_size=128,
              verbose=0,
              callbacks=earlystop)

    return model.evaluate(x_test, y_test, batch_size=128)


# valence arousal
def MLP_classification(getData, getID=None):
    samples, labels, indices = getData()
    if getID is not None:
        _, labels = getID()
    scores = []
    for train, test in indices:
        y_train = keras.utils.to_categorical(labels[train])
        y_test = keras.utils.to_categorical(labels[test])
        scores.append(train_MLP(
            samples[train], y_train, samples[test], y_test))

    # calculating avg loss and accuracy
    avg_acc = 0
    avg_loss = 0
    for loss, acc in scores:
        avg_loss += loss
        avg_acc += acc
    avg_loss /= len(scores)
    avg_acc /= len(scores)

    return avg_loss, avg_acc


DEAP_score = MLP_classification(getDataDEAP)
MAHNOB_score = MLP_classification(getDataMAHNOB_HCI)
DEAP_id_score = MLP_classification(getDataDEAP, getNBDataDEAP)
MAHNOB_id_score = MLP_classification(getDataMAHNOB_HCI, getNBDataMAHNOB_HCI)
MAHNOB_emotion_score = MLP_classification(getEmotionDataMAHNOB_HCI)

print("DEAP valence arousal score:", DEAP_score)
print("MAHNOB valence arousal score:", MAHNOB_score)
print("DEAP subject id score:", DEAP_id_score)
print("MAHNOB subject id score:", MAHNOB_id_score)
print("MAHNOB emotion score:", MAHNOB_emotion_score)
