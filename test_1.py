import pandas as pd

data = pd.read_csv("/home/pyo/Downloads/module4/train_20000.csv", names=['ID','product','a','b'])
data=data.drop(['a','b'],axis =1)


cart = []
ID = data['ID'][0]
temp = []
for i in range(0,20000):
	if(ID==data['ID'][i]):
		temp.append(data['product'][i])
	else:
		cart.append(temp)
		temp=[]
		ID=data['ID'][i]
		temp.append(data['product'][i])


