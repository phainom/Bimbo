# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 18:50:23 2016

by Matthäus Deutsch
Computer Science HU Berlin
Project: "Kaggle Bimbo Challenge"
"""

### Imports
import pandas as pd
import numpy as np
from numpy.random import choice

#%% Methods to load, preprocess and aggregate data

# define global variable giving the spanish and english variable names
names_dict = {'Semana':'WeekNum', 'Agencia_ID':'DepotId', 'Canal_ID':'ChannelId',\
                    'Ruta_SAK':'RouteId', 'Cliente_ID':'ClientId','Producto_ID':'ProductId',\
                    'Venta_uni_hoy':'SalesUnits', 'Venta_hoy':'SalesPesos',\
                    'Dev_uni_proxima':'ReturnsUnits', 'Dev_proxima':'ReturnsPesos', 'Demanda_uni_equil':'AdjDemand',
                    'NombreProducto':'ProductName', 'NombreCliente':'ClientName','Town':'Town','State':'State'}
#path = 'C:/Users/mdeut/Documents/kaggle/Bimbo'
path = '~/kaggle/Bimbo'

def indexes(shape, perc):
    '''
    Given data points return a generator for subset of indices to leave out
    
    Input:
    shape - shape of the data (datapoints x features)
    perc - percentage of datapoints
    '''
    # yield instead of return?    
    out = choice(shape[0], size = int((1-perc/100.0) * shape[0]), replace = False)
    return np.delete(out, np.where(out == 0))


def merge_tt(train, test, values, on):
    '''
    merges additional values with train and testset.
    
    train, test: train, testset
    values: new values to add
    on: column used for merging
    '''
    return train.merge(values, how = 'left', on = on), test.merge(values, how = 'left', on = on)

def get_product_agg(cols, skiprows = None, nrows = None):
    '''
    loads certain colums of the training data
    aggregates the data
    
    Input:
    cols - string names of features - origingal names!
    skiprows - number of rows to skip
    nrows - number of rows to use
    
    Output:
    aggregated dataset containing the count, sum, mean, median, min, max of the products in each weak
    '''
    df_train = pd.read_csv(path + '/Data/train.csv', usecols = ['Semana', 'Producto_ID'] + cols,
                           dtype  = {'Semana': 'int32',
                                     'Producto_ID':'int32',
                                     'Venta_hoy':'float32',
                                     'Venta_uni_hoy': 'int32',
                                     'Dev_uni_proxima':'int32',
                                     'Dev_proxima':'float32',
                                     'Demanda_uni_equil':'int32',
                                     'Agencia_ID':'int32'},
                            skiprows = skiprows, nrows = nrows)
    agg = df_train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(['count','sum', 'min', 'max','median','mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(df_train)
    return agg

def get_train(skiprows = None, nrows = None):
    '''
    loads the training data, english feature names
    
    Input:
    skiprows - number of rows to skip
    nrows - number of rows to use
    '''
    df_train = pd.read_csv(path + '/Data/train.csv',
                       dtype  = {'Semana': 'int32',
                                 'Producto_ID':'int32',
                                 'Venta_hoy':'float32',
                                 'Venta_uni_hoy': 'int32',
                                 'Dev_uni_proxima':'int32',
                                 'Dev_proxima':'float32',
                                 'Demanda_uni_equil':'int32',
                                 'Agencia_ID':'int32'},
                        skiprows = skiprows, nrows = nrows)
    df_train.rename(columns = names_dict, inplace = True)
    return df_train
    
def get_test(skiprows = None, nrows = None):
    '''
    loads the test data, english feature names
    
    Input:
    skiprows - number of rows to skip
    nrows - number of rows to use
    '''
    df_test = pd.read_csv(path + '/Data/test.csv', 
                       dtype  = {'Semana': 'int32',
                                 'Producto_ID':'int32',
                                 'Venta_hoy':'float32',
                                 'Venta_uni_hoy': 'int32',
                                 'Dev_uni_proxima':'int32',
                                 'Dev_proxima':'float32',
                                 'Demanda_uni_equil':'int32',
                                 'Agencia_ID':'int32'},
                        skiprows = skiprows, nrows = nrows)
    df_test.rename(columns = names_dict, inplace = True)
    return df_test
    
def preprocess(skiprows = None, nrows = None):
    '''
    preprocesses the data, joins additional information about the products
    
    mostly taken from
    https://www.kaggle.com/vykhand/grupo-bimbo-inventory-demand/exploring-products
    '''
    
    # load data
    test = get_test(skiprows, nrows)
    train = get_train(skiprows, nrows)
    product = pd.read_csv(path + '/Data/producto_tabla.csv', 
                          dtype = {'Producto_ID':'int32', 'NombreProducto':'object'})
    product.rename(columns = names_dict, inplace = True)
    town = pd.read_csv(path + '/Data/town_state.csv', 
                          dtype = {'Agencia_ID':'int32', 'Town':'object', 'State':'object'})
    town.rename(columns = names_dict, inplace = True)
    client = pd.read_csv(path + '/Data/cliente_tabla.csv',
                          dtype = {'Cliente_ID':'int32', 'NombreCliente':'object'})
    client.rename(columns = names_dict, inplace = True)
    
    # extract short-names, brand, weight and pieces
    product["ShortName"] = product.ProductName.str.extract("^(\D*)", expand=False)
    product["Brand"] = product.ProductName.str.extract("^.+\s(\D+) \d+$", expand=False)
    w = product.ProductName.str.extract('(\d+)(Kg|g)', expand=True)
    product["Weight"] = w[0].astype("float")*w[1].map({"Kg":1000, "g":1})
    product["Pieces"] = product.ProductName.str.extract("(\d+)p", expand=False).astype("float")
    product["Weight"] = product["Weight"].fillna(0)
    product["Pieces"] = product["Pieces"].fillna(1)
    product["WeightperPiece"] = product["Weight"] / product["Pieces"]
    product.drop(["ProductName"], axis=1, inplace=True)
    
    # clean client names
    client = client.drop_duplicates(subset="ClientId")
    client["ClientName"] = client["ClientName"].apply(lambda x: " ".join(x.split()))

    # merging
    train, test = merge_tt(train, test, product, on = 'ProductId')
    train, test = merge_tt(train, test, town, on = 'DepotId')
    train, test = merge_tt(train, test, client, on = 'ClientId')
    
    # splitting training data from target
    train['logDemand'] = np.log1p(train['AdjDemand'])
    target = train[['logDemand','WeekNum']]
    train.drop(['AdjDemand'], inplace = True, axis = 1)
    
    target.to_csv(path + '/Data/target_full.csv', index = False)
    train.to_csv(path + '/Data/train_full.csv', index = False)
    test.to_csv(path + '/Data/test_full.csv', index = False)
    
    print('Data writing done')
    
    return train, test, target