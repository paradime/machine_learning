from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import tree
import numpy as np
iris = load_iris()

train_data, test_data, train_target, test_target = train_test_split(iris['data'], iris['target'],
    random_state=0)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(clf.score(test_data, test_target))
