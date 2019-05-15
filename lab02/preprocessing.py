import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder


def dimenReduceWithLDA(feature_path, label_path):
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

    return samples, labels, label_encoder


def DEAP():
    dimenReduceWithLDA("data/DEAP/EEG_feature.txt", "data/DEAP/valence_arousal_label.txt")


def MAHNOB_HCI():
    dimenReduceWithLDA("data/MAHNOB-HCI/EEG_feature.txt", "data/MAHNOB-HCI/valence_arousal_label.txt")
