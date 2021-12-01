# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:06:32 2021

# Cornershop Data Science Test #

Author: Francisco Mena

In this notebook I design a machine learning model to predict the delivery times,
as part of the application for the data science position at Cornershop


"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
semilla = 2077
np.random.seed(semilla)


#%% Getting familiar with the data
# First, data on the products


dforderprod = pd.read_csv("data\order_products.csv")

dforderprod.dtypes
#checking for nans
dforderprod.isna().sum()

#number of orders: 9978
dforderprod["order_id"].nunique()

#number of products: 15422
dforderprod["product_id"].nunique()

#quantity: up to a 100 units bought, a minimum of 0.05, perhaps refers to weight
dforderprod["quantity"].describe()

dforderprod.loc[dforderprod["quantity"]<1, "buy_unit"].value_counts()
#Indeed, quantities below 1 refer to weight, items bought in bulk

#%%
#%% Now data on orders

dforder = pd.read_csv("data\orders.csv")
dforder.dtypes
dforder.isna().sum()
#2000 nans = test set

dforder["order_id"].duplicated().sum()

#Minimum latitude is in Chiguayante, BioBio
dforder.loc[dforder["lat"].argmin()][["lat", "lng"]]

#Max latitude is in La Serena, Coquimbo
dforder.loc[dforder["lat"].argmax()][["lat", "lng"]]

#Min longitude is in San Pedro de la Paz, BioBio
dforder.loc[dforder["lng"].argmin()][["lat", "lng"]]

#Max longiture is in Barnechea, RM
dforder.loc[dforder["lng"].argmax()][["lat", "lng"]]


dforder["promised_time"] = pd.to_datetime(dforder["promised_time"])
dforder["promised_time"].min()
dforder["promised_time"].max()
#All orders are between 18-20 Oct '19

dforder["on_demand"].value_counts()
dforder["on_demand"].value_counts(normalize = True)
#55% vs 45% on demand

#number of shoppers: 2864
dforder["shopper_id"].nunique()

#476 different stores
dforder["store_branch_id"].nunique()
dforder["store_branch_id"].value_counts()[:10]
dforder["store_branch_id"].value_counts(normalize = True)[:5].sum()
#TFive stores cover over 50% of the orders

#between 12 and 305 minutes (5 hours)
dforder["total_minutes"].describe()

#%% Frequent store or not so common store?

#Let's build a feature on how commonly a store is used
common_store = dforder["store_branch_id"].isin(dforder["store_branch_id"].value_counts(normalize = True)[:5].index)

dforder["common_store"] = common_store
dforder["common_store"] = dforder["common_store"].map({True:1, False:0})

#%% Let's do some time features

plt.figure()
dforder["promised_time"].dt.hour.value_counts().sort_index().plot.bar()
plt.close()
dfaux = dforder.copy()
dfaux["hour"] = dfaux["promised_time"].dt.hour
dforder.dtypes
#No orders between 3am and 11am 

plt.figure()
sns.boxplot(data = dfaux, x = "hour", y = "total_minutes")
plt.close()
#there isn't a strong correlation between time of day and total minutes

def timeofday(t):
    if (t>=22)|(t<4):
        return "night"
    elif (t>=4)&(t<15):
        return "morning_n_lunch"
    elif (t>=15)&(t<19):
        return "afternoon"
    elif (t>=19)&(t<22):
        return "evening"
    else:
        return np.nan

dforder["hourofday"] = dforder["promised_time"].dt.hour + dforder["promised_time"].dt.minute/60 

dforder["timeofday"] = [timeofday(x) for x in dforder["hourofday"]]
dforder[["hourofday", "timeofday"]]
dforder["timeofday"].isna().sum()
dforder["timeofday"].value_counts()
 
#%%
#Now, hour of day is not cyclical,  00:00 follows 24:00, which is not continuous
#Let's make time a cyclical feature
    
dforder["hourofday_x"] = np.sin( 2 * np.pi * dforder["hourofday"] / 24 )
dforder["hourofday_y"] = np.cos( 2 * np.pi * dforder["hourofday"] / 24 )


#%%
#%% Now, data on the shoppers

dfshopper = pd.read_csv("data/shoppers.csv")
dfshopper.isna().sum() #there are seveal nan

dfshopper.dtypes

#no duplicated ids
dfshopper["shopper_id"].duplicated().sum()

dfshopper["shopper_id"].nunique() #2864 shoppers

dfshopper["seniority"].value_counts()
#Only four seniority types

dfshopper["found_rate"].describe() #between 74% and 97%
dfshopper["found_rate"].isna().sum()/len(dfshopper) #3% are nan

dfshopper["picking_speed"].describe() #between .6 and 7.

dfshopper["accepted_rate"].describe() #between 0.24 and 1
dfshopper["accepted_rate"].isna().sum()/len(dfshopper) #1% are nan

dfshopper["rating"].describe() #between 3.8 and 5.0
dfshopper["rating"].isna().sum()/len(dfshopper) #3% are nan

#%% Lets map seniority for clarity

dfshopper["seniority"].value_counts()
#Only four seniority types
mapeo = {"6c90661e6d2c7579f5ce337c3391dbb9":"A",
 "50e13ee63f086c2fe84229348bc91b5b":"B",
 "41dc7c9e385c4d2b6c1f7836973951bf":"C",
 "bb29b8d0d196b5db5a5350e5e3ae2b1f":"D"}

dfshopper["seniority"] = dfshopper["seniority"].map(mapeo)
dfshopper["seniority"].value_counts()


#%% Feature for speed

dfshopper["picking_speed"].describe() #between .6 and 7.
dfshopper["picking_speed_fast"] = dfshopper["picking_speed"]<2

#separate between "slow" and "fast" shoppers
dfshopper["picking_speed_fast"] = dfshopper["picking_speed_fast"].map({True:1, False:0})

#%%
#%% Now data on the stores

dfstore = pd.read_csv("data/storebranch.csv")
dfstore.isna().sum() #no nans

dfstore.dtypes

#no duplicates
dfstore["store_branch_id"].duplicated().sum()
dfstore["store_branch_id"].nunique() #476 stores

dfstore["store_id"].duplicated().sum()  #255 duplicates
dfstore["store_id"].nunique() #221 stores
dfstore["store_id"].value_counts()

#Minimum latitude is in Chiguayante, BioBio
dfstore.loc[dfstore["lat"].argmin()][["lat", "lng"]]

#Max latitude is in La Serena, Coquimbo
dfstore.loc[dfstore["lat"].argmax()][["lat", "lng"]]

#Min longitude is in San Pedro de la Paz, BioBio
dfstore.loc[dfstore["lng"].argmin()][["lat", "lng"]]

#Max longiture is in Barnechea, RM
dfstore.loc[dfstore["lng"].argmax()][["lat", "lng"]]

#change names to merge dataframes
dfstore.rename(columns = {"lat":"lat_store", "lng":"lng_store"}, inplace = True)

#%%
#Merge dataframes

dforder.dtypes
dforderprod.dtypes
dfshopper.dtypes
dfstore.dtypes

df = pd.merge(dforder, dfshopper, how = 'left', left_on = 'shopper_id' ,right_on = 'shopper_id')
df.isna().sum() #nans increase after merge

df = pd.merge(df, dfstore, how = 'left', left_on = 'store_branch_id' ,right_on = 'store_branch_id')
df.isna().sum()


#%% Feature engineering

#Let's build features

#FEATURE: distance between store and delivery
# Eucledian distance would work since stores and deliveries are close,
# but let's be thorough in the calculation anyway
#I'll use this library https://pypi.org/project/haversine/

import haversine as hs
loc1 = [(x,y) for x,y in zip(df["lat"], df["lng"])]
loc2 = [(x,y) for x,y in zip(df["lat_store"], df["lng_store"])]
distance = [hs.haversine(x,y) for x,y in zip(loc1,loc2 )]   #distance in km

df["distance"] = distance

df["distance"].describe()

# The distance the shopper will travel is certainly longer,
# but helps as a metric of how far they needs to travel

#%%
# More info on the locations, let's use geocoder
# https://geopy.readthedocs.io


coords = ["{},{}".format(x, y) for x,y in zip(df["lat"], df["lng"])]
coords = pd.DataFrame(data = coords, columns = ["coordinates"])


    #%%

"""
####### I've already run the following part and saved the results to a file.
####### Since it takes too long to run, I comment it here

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from datetime import datetime

locations = []
place = []
start = datetime.now()
for kk in range(len(coords)):
# for kk in range(80, len(coords)):
# for kk in range(7397, len(coords)):
    
    print(kk)
    
    app = "testapp" + str(kk)
    geolocator = Nominatim(user_agent= app)
    
    
    ans = geolocator.reverse( coords.iloc[kk] )
    
    if "city" in ans.raw['address'].keys():
        place.append("city")
        locations.append(  ans.raw['address']["city"]  )
        
    elif "suburb" in ans.raw['address'].keys():
        place.append("suburb")
        locations.append(  ans.raw['address']["suburb"]  )
    
    elif "town" in ans.raw['address'].keys():        
        place.append("town")
        locations.append(  ans.raw['address']["town"]  )
        
    elif "neighbourhood" in ans.raw['address'].keys():        
        place.append("neighbourhood")
        locations.append(  ans.raw['address']["neighbourhood"]  )
        
    else:
        place.append(np.nan)
        locations.append(np.nan)
    

end = datetime.now()
print(end - start)
# location = geolocator.reverse("{},{}".format(loc2[0][0], loc2[0][1]))
# location
#%%
df_locations = pd.DataFrame({"comuna":locations, "place_type":place[:-1]})
df_locations.isna().sum()
df_locations["comuna"].value_counts().sort_index()
df_locations["place_type"].value_counts()

df_locations.to_csv("comunas_from_coordinates.csv")


#%%
df.columns
coords_store = ["{},{}".format(x, y) for x,y in zip(df["lat_store"], df["lng_store"])]
coords_store = pd.DataFrame(data = coords_store, columns = ["coordinates"])


locations_store = []
place_store = []

start = datetime.now()
#%%

# for kk in range(len(coords_store)):
# for kk in range(636,len(coords_store)):
for kk in range(2041,len(coords_store)):
    
    print(kk)
    
    app = "testapp" + str(kk)
    geolocator = Nominatim(user_agent= app)
    
    
    ans = geolocator.reverse( coords.iloc[kk] )
    
    if "city" in ans.raw['address'].keys():
        place_store.append("city")
        locations_store.append(  ans.raw['address']["city"]  )
        
    elif "suburb" in ans.raw['address'].keys():
        place_store.append("suburb")
        locations_store.append(  ans.raw['address']["suburb"]  )
    
    elif "town" in ans.raw['address'].keys():        
        place_store.append("town")
        locations_store.append(  ans.raw['address']["town"]  )
        
    elif "neighbourhood" in ans.raw['address'].keys():        
        place_store.append("neighbourhood")
        locations_store.append(  ans.raw['address']["neighbourhood"]  )
        
    else:
        place_store.append(np.nan)
        locations_store.append(np.nan)
    

end = datetime.now()
print(end - start)



df_locations_store = pd.DataFrame({"comuna":locations_store, "place_type":place_store})
df_locations_store.isna().sum()
df_locations_store["comuna"].value_counts().sort_index()
df_locations_store["place_type"].value_counts()

df_locations_store.to_csv("Stores_comunas_from_coordinates.csv")

"""

#%% Now one can simply load the results

df_locations = pd.read_csv("comunas_from_coordinates.csv", index_col=0)
df_locations_store = pd.read_csv("Stores_comunas_from_coordinates.csv", index_col=0)

#Let's merge the results
df["comuna_client"] = df_locations["comuna"]
df["comuna_store"] = df_locations_store["comuna"]

#%%

df["comuna_client"].value_counts()
df["comuna_store"].value_counts()

(df["comuna_client"] != df["comuna_store"]).sum()
### All orders and stores are in the same comuna!
### That means I can simply use one feature instead of both

df.columns
df.drop(["comuna_store"], axis = 1, inplace= True)
df.rename(columns = {"comuna_client": "comuna"}, inplace = True)


#Now let's group rare comunas into one same group
df["comuna"].value_counts()
rare_comunas = df["comuna"].value_counts().loc[ df["comuna"].value_counts()<100  ].index

df["comuna"] = ["rare_comuna" if x in rare_comunas else x for x in df["comuna"]]


#%% Now for the order products

# first separate between bulk and unit items
dforderprod.dtypes
dforderprod["product_id"].nunique()
dforderprod["order_id"].nunique()
num_unique_products = dforderprod.groupby("order_id")["product_id"].nunique() #number of products per order


dfbulk = dforderprod.loc[dforderprod["buy_unit"] == "KG"].copy()
dfunit = dforderprod.loc[dforderprod["buy_unit"] == "UN"].copy()

#Features based on how much is bought
tot_weight = dfbulk.groupby("order_id")["quantity"].sum() #weight bought
tot_weight.describe()
num_tot_products = dfunit.groupby("order_id")["quantity"].sum() #weight bought
num_tot_products.describe()


#%%
#Frequency of items bougth
plt.figure()
dforderprod["product_id"].value_counts().plot.box()
plt.ylim(0,10)
plt.close()

dforderprod["product_id"].value_counts(normalize = True)

item_data = dforderprod["product_id"].value_counts()
item_data.describe()

#Let's classify items according to frequency bought

item_data = item_data.to_frame("frequency")
item_data.loc[item_data["frequency"]<=8]
item_data.loc[item_data["frequency"]>8]
item_data.loc[item_data["frequency"]<=8, "freq_type"] = "FREQ_rare"
item_data.loc[item_data["frequency"]>8, "freq_type"] = "FREQ_normal"
item_data["freq_type"].value_counts()
item_data.rename_axis("product_id", axis = 0, inplace = True)

dforderprod.columns
dforderprod = pd.merge(dforderprod, item_data, how = "left", left_on = "product_id", right_on = "product_id")

dforderprod.loc[dforderprod["buy_unit"]=="KG", "quantity"].describe()

#%%
###########
"""
Now how much we buy per product, too much? too little?
"""

#UNIT STUFF
median_quantity_bought = dfunit.groupby("product_id")["quantity"].median()
median_quantity_bought.describe()
median_quantity_bought.loc[median_quantity_bought<2]
median_quantity_bought.loc[median_quantity_bought>2]
median_quantity_bought.loc[median_quantity_bought>4]

median_quantity_bought = median_quantity_bought.to_frame("quantity_bought")
median_quantity_bought.loc[median_quantity_bought["quantity_bought"]<2, "amount_bought"] = "Q_a_little"
median_quantity_bought.loc[(median_quantity_bought["quantity_bought"]>=2) & (median_quantity_bought["quantity_bought"]<4), "amount_bought"] = "Q_some"
median_quantity_bought.loc[median_quantity_bought["quantity_bought"]>=4, "amount_bought"] = "Q_tons"
median_quantity_bought["amount_bought"].value_counts()
median_quantity_bought["amount_bought"].value_counts(normalize = True)


#BULK STUFF
median_bulk_bought = dfbulk.groupby("product_id")["quantity"].median()
median_bulk_bought.describe()
median_bulk_bought.loc[median_bulk_bought<1].count()
median_bulk_bought.loc[(median_bulk_bought>=1)&((median_bulk_bought<2))].count()
median_bulk_bought.loc[median_bulk_bought>2].count()

median_bulk_bought = median_bulk_bought.to_frame("quantity_bought")
median_bulk_bought.loc[median_bulk_bought["quantity_bought"]<1, "amount_bought"] = "Q_a_little"
median_bulk_bought.loc[(median_bulk_bought["quantity_bought"]>=1) & (median_bulk_bought["quantity_bought"]<2), "amount_bought"] = "Q_some"
median_bulk_bought.loc[median_bulk_bought["quantity_bought"]>=2, "amount_bought"] = "Q_tons"
median_bulk_bought["amount_bought"].value_counts()
median_bulk_bought["amount_bought"].value_counts(normalize = True)


stuff_bought = pd.concat([median_quantity_bought, median_bulk_bought])

dforderprod = pd.merge(dforderprod, stuff_bought[["amount_bought"]], how = "left", left_on = "product_id", right_on = "product_id")

#%%
dforderprod.dtypes
dforderprod["amount_bought"].value_counts()
dforderprod["amount_bought"].value_counts(normalize = True)
#70% of the products bought are one item or low weight. 
#About 20% are two to 4 items, or less than 2 kg, and 9% are several items or over 2 kg

#%%

#type of items bought per order
item_type = pd.crosstab(index = dforderprod["order_id"], columns = dforderprod["buy_unit"])
item_type.isna().sum()

#how frequently bought are the items in the order
item_freq = pd.crosstab(index = dforderprod["order_id"], columns = dforderprod["freq_type"])
item_freq.isna().sum()

item_amount = pd.crosstab(index = dforderprod["order_id"], columns = dforderprod["amount_bought"])
item_amount.isna().sum()

#%%
#Now that we have features for the type of products bought, we can merge it to the main df

df["order_id"].duplicated().sum()
df.head()
df = pd.merge(df, item_type, how = 'left', left_on = "order_id", right_on = "order_id")
df.isna().sum()

df = pd.merge(df, item_freq, how = 'left', left_on = "order_id", right_on = "order_id")
df.isna().sum()

df = pd.merge(df, item_amount, how = 'left', left_on = "order_id", right_on = "order_id")
df.isna().sum()


#%% Now let's remove the features that we'll not use for modeling

todrop = ["promised_time", "shopper_id", "store_branch_id", "hourofday", "store_id"]

df.drop(todrop, axis = 1, inplace = True)

df.set_index("order_id", inplace = True)


#%% separate the data that will be used for training and for the final prediction

finaltest = df.loc[df["total_minutes"].isna()]
data = df.loc[~df["total_minutes"].isna()]

#%%

#Drop the files with NaNs
data.isna().sum()
data.dropna(axis = 0, inplace = True)

#%% SHUFFLE
# Just in case let's shuffle the data, to make sure we're not giving the model a bias hidden in the data

data = data.sample(frac=1, random_state = semilla)

#%% Now let's prepare for modeling!


dataOG = data.copy() #to use later

data.columns
y = data.pop("total_minutes")
X = data.copy()


#%%

y.plot.box()

#There are several outliers, with time over the ~170 min mark.
#Models with regulatization like ElasticNet will be useful.


#%%

#Let's look at correlations
corr = X.corr().abs()

plt.figure(figsize = (6,6))
sns.heatmap(corr)
plt.show()

#coordinates are correlated, which makes sense, and frequencies and amounts are, wich is also reasonable
#this indicates that correlation won't affect regression

#%% Let's process categorical features

from sklearn.preprocessing import OneHotEncoder

catvars = X.dtypes.loc[X.dtypes == object].index

for var in catvars:
    # print(duf[var].value_counts())

    X[var] = [var + '_'  + x for x in X[var]]

    ohe = OneHotEncoder()
    A = pd.DataFrame(  ohe.fit_transform(X[[var]]).toarray(), columns = ohe.categories_[0] , index = X.index)

    X = X.merge(A[ohe.categories_[0]], left_index = True, right_index = True)


X.columns
X.drop(catvars, axis = 1, inplace = True)
# X.drop(["timeofday","seniority"], axis = 1, inplace = True)


#%% Lets choose a few models to try

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

models = [("elastic_net", ElasticNet(random_state=semilla) ),
          ("RF", RandomForestRegressor(random_state=semilla, criterion="mae", n_jobs = -1)),
          ("svr", SVR())
          ]


#%% Now let's do a K-Fold cross validation, to avoid overfitting due to choosing a specific split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone

kf = KFold(n_splits=3)

for name, model in models:

    mae = []
    mse = []

    for train_index, test_index in kf.split(X):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        scale = StandardScaler()
        X_train_scaled = scale.fit_transform(X_train)
        X_test_scaled = scale.transform(X_test)
        
        reg = clone(model)
        reg.fit(X_train_scaled, y_train)
        ypred = reg.predict(X_test_scaled)
        
        mae.append( mean_absolute_error(y_test, ypred) )
        mse.append( mean_squared_error(y_test, ypred, squared=False) )
        
    print("MAE for " + name + ": mean is " + str(np.mean(mae)) + " and std " + str(np.std(mae)))
    print("RMSE for " + name + ": mean is " + str(np.mean(mse)) + " and std " + str(np.std(mse)))
    print("\n")

#EN 18.8/24.8
#RF 17.3/23
#SVR 19/26
    
#%% md
"""
Since the std of all models is very small, less than a minute, there's no concern of overfitting
due to choosing a specific split. Having checked that, we can use a more powerful technique:
    Let's use gridsearch to find the optimal parameters
"""


#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=semilla)

y_train.describe()
y_test.describe()


#%% Scaling

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

#%%

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


#We use mae to avoid the outliers from affecting the training too much
reg = RandomForestRegressor(random_state=semilla, criterion = "mae")

# Number of trees in random forest
n_estimators = [10, 100, 200]
# Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 20, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
params = {'n_estimators': n_estimators,
               # 'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


grid = GridSearchCV(estimator = reg, param_grid = params, 
                    scoring = "neg_mean_absolute_error", 
                    cv = 3, n_jobs = 15, verbose = 2)

#%%
print("Working on gridsearch")

grid.fit(X_train_scaled, y_train)


#%% 
grid.best_params_

"""
{'bootstrap': True,
 'max_depth': 20,
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 200}

"""
#%%
reg = grid.best_estimator_

ypred = reg.predict(X_test_scaled)

print( mean_absolute_error(y_test, ypred) )
print( mean_squared_error(y_test, ypred, squared=False) )

#new values
# 16.6366439997288
# 22.830708375098602

#%%        
        
# Now SVR gridsearch

from sklearn.svm import SVR
gsc = GridSearchCV(
        estimator=SVR(),
        param_grid={'kernel': ["rbf", "linear"],
            'C': [1, 100],
            'epsilon': [0.001, 1, 5],
            'gamma':["auto", "scale"]
        },
        cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=15)

   
    
gsc.fit(X_train_scaled, y_train)

        
#%%

gsc.best_params_        
"""
{'C': 100, 'epsilon': 5, 'gamma': 'auto', 'kernel': 'rbf'}
"""

#%%
reg = gsc.best_estimator_

ypred = reg.predict(X_test_scaled)

print( mean_absolute_error(y_test, ypred) )
print( mean_squared_error(y_test, ypred, squared=False) )

# 16.924572894720576
# 23.194546487539633
#%%
from sklearn.linear_model import ElasticNetCV

reg = ElasticNetCV(l1_ratio= [.1, .5, .7, .9, .95, .99, 1], random_state = semilla )

reg.fit(X_train_scaled, y_train)
ypred = reg.predict(X_test_scaled)

print(reg.l1_ratio_)

print( mean_absolute_error(y_test, ypred) )
print( mean_squared_error(y_test, ypred, squared=False) )
# 17.880412062299204
# 24.09595343713386

#%% md

# Now let's try with ligthgbm
# lightgbm handles categorial features, so let's avoid using one hot encoding

#%%
import lightgbm as lgb
# from lightgbm import LGBMRegressor
print(lgb.__version__)


dataOG.columns

catvars = dataOG.dtypes.loc[dataOG.dtypes == object].index

for var in catvars:

    dataOG[var] = dataOG[var].astype('category')
 
#%%

label = dataOG.pop("total_minutes")
Xlgb = dataOG.copy()


Xlgb["on_demand"] = Xlgb["on_demand"].astype(int)

X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(Xlgb, label, test_size=0.2, random_state=semilla)

#%%

train_data = lgb.Dataset(X_train_lgb, label = y_train_lgb)
eval_data = lgb.Dataset(X_test_lgb, label = y_test_lgb, reference = train_data)

X_test_lgb.columns.shape
X_train_lgb.columns.shape
X_test_lgb.dtypes

#%% Train


parameters = {
        'objective': 'regression',
        'metric': 'mae',               
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_cat_to_onehot': 10,
        'bagging_freq': 10,
        'feature_fraction': 0.9,
        'n_jobs':9, 
        'seed': semilla
    }

num_round = 1000
reg = lgb.train(params=parameters, train_set=train_data, 
                num_boost_round=num_round, 
                valid_sets=eval_data, 
                early_stopping_rounds=5,
                verbose_eval=0)


y_pred = reg.predict(X_test_lgb, num_iteration=reg.best_iteration)

print( mean_absolute_error(y_test, ypred) )
print( mean_squared_error(y_test, ypred, squared=False) )

# 17.880412062299204
# 24.09595343713386

#%% Let's try with hyperparameter tuning




model = lgb.LGBMRegressor(random_state=semilla)
param_dists = {
    "n_estimators": [400, 700, 1000],
    "colsample_bytree": [0.7, 0.8, 1],
    "max_depth": [-1, 15, 20, 25],
    "learning_rate":[0.01, 0.001],
    "num_leaves": [10, 100, 1000, 3000],
    "min_data_in_leaf":[100, 1000],
    "reg_alpha": [1.1, 1.2, 1.3],
    "reg_lambda": [1.1, 1.2, 1.3],
    "min_split_gain": [0., 1., 15.],
    "subsample": [0.7, 0.8, 0.9],
    "subsample_freq": [20],
    "random_state":[semilla]
}

gs = RandomizedSearchCV(estimator = model, param_distributions=param_dists, 
                  n_iter=100,
                  scoring="neg_mean_absolute_error", 
                  n_jobs=10,
                  cv=3,
                  random_state=semilla,
                  verbose=2
                  )
gs.fit(X_train_lgb, y_train_lgb)


ypred = gs.predict(X_test_lgb)

print( mean_absolute_error(y_test, ypred) )
print( mean_squared_error(y_test, ypred, squared=False) )

# 15.896987017506243
# 21.591355890477775

#%%
# This is the best model so far

#%%

gs.best_params_

"""
{'subsample_freq': 20,
 'subsample': 0.9,
 'reg_lambda': 1.1,
 'reg_alpha': 1.3,
 'random_state': 2077,
 'num_leaves': 1000,
 'n_estimators': 1000,
 'min_split_gain': 1.0,
 'min_data_in_leaf': 100,
 'max_depth': 15,
 'learning_rate': 0.01,
 'colsample_bytree': 0.7}
"""
#%% Let's explain the model

#First random forest model
rf = grid.best_estimator_

import shap

explainer = shap.Explainer(rf)
shap_values = explainer(X_test_scaled)

#%%
shap.summary_plot(shap_values, X_test_scaled, feature_names=X_test.columns, )   


#%%

ax = lgb.plot_importance(reg, max_num_features=15)

#%%
"""
It's interesting to notice that both models share similar features among the most important ones.
For instance, for both models the picking speed, distance, hour of day, frequency that a store is bought from
and quantity/amount of items bought are very important. 
It's also good to notice that the features I created were useful to the model, such as cyclical time,
distance, comuna, frequency of stores, and quantity of items bought.


Now that we have the best model: lightgbm, let's use it to predict on the set we need to submit
"""



#%%


catvars = finaltest.dtypes.loc[finaltest.dtypes == object].index

for var in catvars:

    finaltest[var] = finaltest[var].astype('category')

#%%

finaltest.dtypes

total_minutes = finaltest.pop("total_minutes")

finaltest.columns

total_minutes_pred = gs.predict(finaltest)

total_minutes = total_minutes.to_frame("total_minutes_predicted")
total_minutes["total_minutes_predicted"] = total_minutes_pred
total_minutes.to_csv("total_minutes_predicted.csv")
total_minutes.to_excel("total_minutes_predicted.xlsx")

#%%