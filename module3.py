import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train = pd.read_csv('./Downloads/train.csv')
test = pd.read_csv('./Downloads/test.csv')


train_ID=train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

train['SalePrice']=np.log1p(train['SalePrice'])

train = train.drop(train[(train['BsmtFinSF2']>1200)].index)
train = train.drop(train[(train['TotalBsmtSF']>3000)].index)
train = train.drop(train[(train['GarageArea']>1200)].index)
train = train.drop(train[(train['EnclosedPorch']>4000)].index)
train = train.drop(train[(train['MiscVal']>6000)].index)
train = train.drop(train[(train['OpenPorchSF']>500)].index)
train = train.drop(train[(train['EnclosedPorch']>4000)].index)
train = train.drop(train[(train['LotFrontage']>200)].index)
train = train.drop(train[(train['LotArea']>100000)].index)
train = train.drop(train[(train['MasVnrArea']>1500)].index)
train = train.drop(train[(train['BsmtFinSF1']>2500)].index)


ntrain =train.shape[0]
ntest= test.shape[0]

y_train= train.SalePrice.values


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)



from sklearn import preprocessing
all_data_2 =all_data.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
all_data_2 = all_data_2.apply(le.fit_transform)
for i in all_data_2:
	all_data[i]=all_data_2[i]



all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]


from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)


all_data = pd.get_dummies(all_data)


train = all_data[:ntrain]
test = all_data[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=720,
                              max_bin = 100, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_est  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
					max_depth=1, random_state=0, loss='ls')

model_est.fit(train, y_train)

est_train_pred=model_est.predict(train)
est_pred=np.exp(model_est.predict(test))

model_regr = RandomForestRegressor(bootstrap=True, criterion='mse',
				max_depth=2,max_features='auto',max_leaf_nodes=None,
				min_samples_leaf=1,min_samples_split=2,
				min_weight_fraction_leaf=0.0, n_estimators=10,
				n_jobs=1,oob_score=False, random_state=0,
				verbose=0, warm_start=False)
model_regr.fit(train, y_train)


regr_train_pred=model_regr.predict(train)
regr_pred=np.exp(model_regr.predict(test))


model_lgb.fit(train, y_train)

lgb_train_pred=model_lgb.predict(train)
lgb_pred = np.exp(model_lgb.predict(test))

from sklearn.metrics import mean_squared_error 
from math import sqrt

rmse = sqrt(mean_squared_error(y_train.astype(int),regr_train_pred[:ntrain].astype(int)))
print('random forest rmse =' ,rmse)


rmse = sqrt(mean_squared_error(y_train.astype(int),est_train_pred[:ntrain].astype(int)))
print ('gradiant boostin rmse = ',rmse)



rmse = sqrt(mean_squared_error(y_train.astype(int),lgb_train_pred[:ntrain].astype(int)))
print ('lightgbm rmse = ',rmse) 


from sklearn import linear_model
lr = linear_model.LinearRegression()
model_lr = lr.fit(train,y_train)

lr_train_pred = model_lr.predict(train)
lr_pred = np.exp(model_lr.predict(test))

rmse = sqrt(mean_squared_error(y_train.astype(int),lr_train_pred[:ntrain].astype(int)))
print('linear regression rmse =' ,rmse)


ensemble = lgb_train_pred*0.9+lr_train_pred*0.1
rmse = sqrt(mean_squared_error(y_train.astype(int),ensemble[:ntrain].astype(int)))
print('ensemble rmse = ', rmse)

ensemble_pred = lgb_pred*0.7+regr_pred*0.3

submission=pd.DataFrame()
submission['Id']=test_ID
submission['SalePrice']= ensemble_pred
 
submission.to_csv('submission7r3.csv', index=False)



