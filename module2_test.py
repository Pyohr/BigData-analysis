from sklearn import tree
clf = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
from sklearn import datasets
iris=datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=0)
y_pred=clf.fit(x_train,y_train).predict_proba(x_test)
print(y_pred)

print(y_pred[1][0])
print(y_pred[1][1])
print(y_pred[1][2])

print(y_pred[0])
print(y_pred[1])

import pandas as pd
import numpy as np

test = {'first flower' : ['flower name1',y_pred[0][1]],
	'second flower' : ['flower name',y_pred[1][1]]}
print(test)

df = pd.DataFrame(data=test)
df.to_csv('test.csv')

m1 = max(y_pred[1][0],y_pred[1][1])
print(m1)

sorted(y_pred, key=lambda x:x[0])
print(y_pred)


