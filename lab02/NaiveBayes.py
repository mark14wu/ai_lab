from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from preprocessing import getNBDataMAHNOB_HCI, getNBDataDEAP

# clf = BernoulliNB()
clf = GaussianNB()

# feed DEAP feature and id
(samples, _, indices), id_list = getNBDataDEAP()
scores = cross_val_score(clf, samples, id_list, cv=indices)
print("DEAP subject id:", scores.mean())

# feed MAHNOB feature and id
(samples, labels, indices), id_list = getNBDataMAHNOB_HCI()
scores = cross_val_score(clf, samples, id_list, cv=indices)
print("MAHNOB-HCI subject id:", scores.mean())

# feed MAHNOB emotion label
clf = MultinomialNB()
scores = cross_val_score(clf, samples, labels, cv=indices)
print("MAHNOB-HCI emotion category:", scores.mean())
