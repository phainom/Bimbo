# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:44:39 2016

by Matthäus Deutsch
Computer Science HU Berlin
Project: "Kaggle Bimbo Challenge"
"""

import numpy as np
import data


#%% Simple Mean-based Modell

# define the logmean    
def logmean(x):
    return np.exp(np.mean(np.log(x+1)))-1

# load the training data
print('load training data')
df_train = data.get_train(nrows = 10000)

# compute the means for different configurations
print('compute means')
mean_tab = df_train.groupby('ProductId').agg({'AdjDemand': logmean})
mean_tab2 = df_train.groupby(['ProductId', 'ClientId']).agg({'AdjDemand': logmean})
global_mean = logmean(df_train['AdjDemand'])


# generate estimation for each ProductID-ClientID-pair
def estimate(key):
    key = tuple(key) # key needs to be a tuple
    try:
        est = mean_tab2.at[key,'AdjDemand']
    except KeyError:
        try :
            est = mean_tab.at[key[0],'AdjDemand']
        except KeyError:
            est = global_mean
    return est


# load the test data
print('load test data')
df_test = data.get_test(nrows=10000)
print('compute predictions')
df_test['Demanda_uni_equil'] = df_test[['ProductId', 'ClientId']].\
                apply(lambda x:estimate(x), axis=1)
df_submit = df_test[['id', 'Demanda_uni_equil']]
print(df_submit.shape)
df_submit = df_submit.set_index('id')
df_submit.to_csv('naive_product_client_logmean.csv')
