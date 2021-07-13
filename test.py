from sklearn import tree
clf =tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd 
df1 = pd.read_cdv("./Downloads/play_tennis.csv",header=None)
from sklearn import prepocessing
df1_2 = df1.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
df1_2 = df1_2.apply(le.fit_transform)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
y_pred=clf.fit(X_train,y_train).predict(X_test)
print("Number of mislabled points out of a total %d points :%d" % (X_test.shape[0],(y_test!=y_pred).sum()))
