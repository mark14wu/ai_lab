import warnings

warnings.filterwarnings("ignore")

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.model_selection import cross_val_score

from preprocessing import getDataDEAP, getDataMAHNOB_HCI



def svm_tuning(C_start, C_end, step):
    """
    tune a best C for certain dataset
    """
    def train_model():
        score_list = {}
        for current_C in tqdm(C_list[1:]):
            rbf_svc = SVC(C=current_C, kernel='rbf')
            current_score = cross_val_score(rbf_svc, samples, labels, cv=indices)
            score_list[current_C] = current_score
        return score_list

    C_list: ndarray = np.arange(C_start, C_end, step)

    # training on DEAP
    samples, labels, indices = getDataDEAP()
    print("Working on DEAP dataset")
    deap_result = train_model()

    # training on MAHNOB
    samples, labels, indices = getDataMAHNOB_HCI()
    print("Working on MAHNOB-HCI dataset")
    mahnob_result = train_model()

    # calculating best C and best score
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
    # read data
    samples, labels, indices = getData()
    # build model
    svc = SVC(C=C, kernel='rbf')
    # get cross validation score
    scores = cross_val_score(svc, samples, labels, cv=indices)
    print("score:", scores.mean())


# svm_tuning(0, 50, 0.01)
# svm_tuning(0, 5, 1)

C = 0.81
print("test on DEAP!")
svm_test(C, getDataDEAP)
print("test on MAHNOB-HCI!")
svm_test(C, getDataMAHNOB_HCI)
