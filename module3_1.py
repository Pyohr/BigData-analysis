import pandas as pd
import numpy as np

train = pd.read_csv('./Downloads/train.csv')
test = pd.read_csv('./Downloads/test.csv')

import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

target = np.log(train.SalePrice)

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

train = train[train['GarageArea'] < 1200]
























