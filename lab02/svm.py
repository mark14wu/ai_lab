from sklearn import svm
from sklearn.pipeline import Pipeline

from preprocessing import DEAP, MAHNOB_HCI

rbf_svc = svm.SVC(C=1.0, kernel='rbf', loss="hinge")

svm_clf = Pipeline([()])
from sklearn import datasets
import numpy as np

from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
])
svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])

out:array([1.])
---------------------
作者：菜鸟知识搬运工
来源：CSDN
原文：https: // blog.csdn.net / qq_30815237 / article / details / 88251342
版权声明：本文为博主原创文章，转载请附上博文链接！