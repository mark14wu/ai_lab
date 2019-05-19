from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from preprocessing import getDataDEAP, getDataMAHNOB_HCI

clf = DecisionTreeClassifier()

samples, labels, indices = getDataDEAP()
deap_score = cross_val_score(clf, samples, labels, cv=indices)
print("DEAP:", deap_score.mean())

samples, labels, indices = getDataMAHNOB_HCI()
mahnob_score = cross_val_score(clf, samples, labels, cv=indices)
print("MAHNOB:", mahnob_score.mean())
