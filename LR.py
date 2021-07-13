import numpy as np 
import pandas as pd

X = pd.read_csv("./X.csv",header=None)
X = np.array(X.as_matrix()) 
y = pd.read_csv("./y.csv",header=None) 
y = np.array(y.as_matrix())
np.random.seed(7)

from keras.models import Sequential 
from keras.layers import Dense 

model = Sequential() 
model.add(Dense(20, input_dim=10, kernel_initializer='normal',activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu')) 
model.add(Dense(1)) 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 

model.fit(X, y, epochs=100, batch_size=10) 
y_pred = model.predict(X) 

from sklearn.metrics import mean_squared_error
from math import sqrt 

rmse = sqrt(mean_squared_error(y, y_pred))
print ('rmse = ',rmse) 

