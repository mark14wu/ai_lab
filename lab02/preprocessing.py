import os
import pickle

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def NB_raw_sample(feature_path, label_path):
    with open(feature_path) as feature_file:
        samples = [sample_line.split('\t') for sample_line in feature_file.readlines()]
        samples = np.array(samples, dtype=float)

    with open(label_path) as label_file:
        labels = [int(raw_label) for raw_label in label_file.readlines()]

    labels = np.array(labels)
    return samples, labels

def dimReduceWithLDA(feature_path, label_path):
    # read features from file
    with open(feature_path) as feature_file:
        samples = [sample_line.split('\t') for sample_line in feature_file.readlines()]

    # divide EEG into 5 frequency bands
    samples = [(sample[:32], sample[32:64], sample[64:96], sample[96:128], sample[128:]) for sample in samples]

    # read labels from file
    with open(label_path) as label_file:
        labels = [str(tuple([x.strip() for x in raw_label.split('\t')])) for raw_label in label_file.readlines()]

    # use label encoder
    label_encoder = LabelEncoder()
    # transcode labels like (1, 2) into label arrays
    labels = label_encoder.fit_transform(labels)

    # generate freq band vectors
    theta = []
    slow_alpha = []
    alpha = []
    beta = []
    gamma = []

    # classify data segments into their respective vectors
    for _theta, _slow_alpha, _alpha, _beta, _gamma in samples:
        theta.append(_theta)
        slow_alpha.append(_slow_alpha)
        alpha.append(_alpha)
        beta.append(_beta)
        gamma.append(_gamma)

    # form a list of numpy arrays (samples)
    samples = [theta, slow_alpha, alpha, beta, gamma]
    samples = [np.array(wave, dtype=float) for wave in samples]

    # use Linear Discriminant Analysis to reduce the dimension (from 32 dims to 3 dims)
    samples = [LDA(n_components=3).fit_transform(wave, labels) for wave in samples]

    # concatenate 5 different frequency bands into a 5 x 3 = 15 dim feature vector
    samples = np.concatenate(samples, axis=1)

    return samples, labels


def getDataDEAP():
    return pickle.load(open("data/preprocessed/svm_deap", 'rb'))

def getDataMAHNOB_HCI():
    return pickle.load(open("data/preprocessed/svm_mahnob_hci", 'rb'))


def getNBDataDEAP():
    id_file = "data/DEAP/subject_video.txt"
    id_list = np.array([int(line.split('\t')[0]) for line in open(id_file).readlines()])
    return pickle.load(open("data/preprocessed/bernoulliNB_deap_hci", 'rb')), id_list


def getNBDataMAHNOB_HCI():
    id_file = "data/MAHNOB-HCI/subject_video.txt"
    id_list = np.array([int(line.split('\t')[0]) for line in open(id_file).readlines()])
    return pickle.load(open("data/preprocessed/bernoulliNB_mahnob_hci", 'rb')), id_list


def getEmotionDataMAHNOB_HCI():
    return pickle.load(open("data/preprocessed/bernoulliNB_mahnob_hci", 'rb'))


def save_data(id_file: str, dest_file: str, samples, labels) -> None:
    if not os.path.exists(dest_file):
        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=114514)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        # read patient id
        id_list = [int(line.split('\t')[0]) for line in open(id_file).readlines()]
        id_list = np.array(id_list)
        # generate cross validation indices
        indices = [(train, test) for train, test in skf.split(samples, id_list)]
        # save to file
        pickle.dump((samples, labels, indices), open(dest_file, 'wb'))

def process():
    # mkdir
    if not os.path.exists("data/preprocessed"):
        os.mkdir("data/preprocessed")

    # save DEAP data for svm
    samples, labels = dimReduceWithLDA("data/DEAP/EEG_feature.txt", "data/DEAP/valence_arousal_label.txt")
    save_data("data/DEAP/subject_video.txt", "data/preprocessed/svm_deap",
              samples, labels)

    # save MAHNOB data for svm
    samples, labels = dimReduceWithLDA("data/MAHNOB-HCI/EEG_feature.txt", "data/MAHNOB-HCI/valence_arousal_label.txt")
    save_data("data/MAHNOB-HCI/subject_video.txt", "data/preprocessed/svm_mahnob_hci",
              samples, labels)

    # save DEAP data for Naive Bayes
    samples, labels = NB_raw_sample("data/DEAP/EEG_feature.txt", "data/MAHNOB-HCI/EEG_emotion_category.txt")
    # samples, labels = dimReduceWithLDA("data/DEAP/EEG_feature.txt", "data/MAHNOB-HCI/EEG_emotion_category.txt")
    save_data("data/DEAP/subject_video.txt", "data/preprocessed/bernoulliNB_deap_hci",
              samples, labels)

    # save MAHNOB data for Naive Bayes
    samples, labels = NB_raw_sample("data/MAHNOB-HCI/EEG_feature.txt", "data/MAHNOB-HCI/EEG_emotion_category.txt")
    # samples, labels = dimReduceWithLDA("data/MAHNOB-HCI/EEG_feature.txt", "data/MAHNOB-HCI/EEG_emotion_category.txt")
    save_data("data/MAHNOB-HCI/subject_video.txt", "data/preprocessed/bernoulliNB_mahnob_hci",
              samples, labels)


process()
