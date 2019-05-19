import warnings

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.svm import SVC
from tqdm import tqdm

from preprocessing import getDataDEAP, getDataMAHNOB_HCI

warnings.filterwarnings("ignore")


def svm_tuning(C_start, C_end, step):
    def train_model():
        score_list = {}
        for current_C in tqdm(C_list[1:]):
            rbf_svc = SVC(C=current_C, kernel='rbf')
            current_scores = []
            for train, test in indices:
                rbf_svc.fit(samples[train], labels[train])
                current_scores.append(float(rbf_svc.score(samples[test], labels[test])))
            current_score = sum(current_scores) / len(current_scores)
            score_list[current_C] = current_score
        return score_list

    C_list: ndarray = np.arange(C_start, C_end, step)

    samples, labels, indices = getDataDEAP()
    print("Working on DEAP dataset")
    deap_result = train_model()
    samples, labels, indices = getDataMAHNOB_HCI()
    print("Working on MAHNOB-HCI dataset")
    mahnob_result = train_model()

    max_C = 0
    max_score = 0
    for C in C_list[1:]:
        current_score = deap_result[C] * mahnob_result[C]
        if current_score > max_score:
            max_score = current_score
            max_C = C
    print("max_c:%f, max_score:%f" % (max_C, max_score))


def svm_test(C, getData):
    """

    :rtype: None
    """
    # test on DEAP
    # X_train, y_train, X_test, y_test, label_encoder = getDataDEAP()
    samples, labels, indices = getData()
    scores = []
    svc = SVC(C=C, kernel='rbf')
    for train, test in indices:
        svc.fit(samples[train], labels[train])
        scores.append(float(svc.score(samples[test], labels[test])))
    print("score: %f" % (sum(scores) / len(scores)))


# svm_tuning(0, 50, 0.01)
# svm_tuning(0, 5, 1)

C = 0.81
print("test on DEAP!")
svm_test(C, getDataDEAP)
print("test on MAHNOB-HCI!")
svm_test(C, getDataMAHNOB_HCI)
