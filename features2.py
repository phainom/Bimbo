# -*- coding: utf-8 -*-
'''
Created on Sun Jul 24 20:10:08 2016

by Matthäus Deutsch
Computer Science HU Berlin
Project: 'Kaggle Bimbo Challenge'

Build features for the first xgb model
'''

from data import merge_tt


# path = 'C:/Users/mdeut/Documents/kaggle/Bimbo'
path = '~/kaggle/Bimbo'

def split(df, weeks=[3, 4, 5]):
    '''
    splits the training data in several weeks
    
    data - data
    weeks - weeknumber to split the data example:
    '''
    select = df['WeekNum'].isin(weeks)
    return df[select], df[~select]
    
def mean_calc(train, test, col):
    '''
    calculates the mean over col using train
    merges the result with train, test    
    col - LIST of strings containing the variable names
    
    
    first value of col has to be the value of mean-calculation
    '''
    if len(col) > 1:
        mean = train[col].groupby(col[1:]).mean()
        mean.columns = [str(col[1:])+'-Mean']
        mean = mean.reset_index()
    else:
        raise KeyError('col needs to consist of various Strings')
    return merge_tt(train, test, mean, on = col[1:])
    
def median_week(train, test, col, week):
    '''
    calculates the median over col of a certain week
    merges the result with train, test
    col - LIST of strings containing the variable names
    '''
    tempData = train[train.WeekNum == week]
    if len(col) > 1:
        median = tempData[col].groupby(col[1:]).median()
        median.columns = [str(col[1:])+'-MedianWeek']
        median = median.reset_index()
    else:
        raise KeyError('col needs to consist of various Strings')
    return merge_tt(train, test, median, on = col[1:])

    
def mult_calc(train, test, col, put = [0, 0, 0], measures = ['mean', 'median', 'count']):
    '''
    multiple calculations for feature col
    
    Inputs:
    train, test - dataset
    col - String consisting of feature name
    put - list putative values for missing values for each measure, needs to be a list of the same size as measures
    measures - string names of functions to compute on logDemand    
    
    Output:
    merged and adjusted train and test set
    '''
    # categorial NA variable
    train[col + 'NA'] = train[col].apply(lambda x: 1 if x == -1 else 0)
    test[col + 'NA'] = test[col].apply(lambda x: 1 if x == -1 else 0)
    
    # compute mean, median, count
    tempData = train[[col, 'logDemand']].groupby(col).agg(measures).logDemand
    tempData = tempData.reset_index()
    names = {}
    for func in measures:
        names[func] = col + func.capitalize()
        
    tempData = tempData.rename(columns = names)
    train, test = merge_tt(train, test, tempData, on = col)
    
    # impute missing values        
    for k in xrange(len(measures)):
        train[names[measures[k]]] = train[names[measures[k]]].fillna(put[k])
        
    return train, test
    
def drop_feat(train, test):
    ''' 
    drops features not both in train and test set
    '''
    for feat in list(set(train.columns) - set(test.columns)):
        del train[feat]
    for feat in list(set(test.columns) - set(train.columns)):
        del test[feat]
    return train, test

    
def build_feat(train, test, name = 'features'):
    '''
    builds various features for boosting
    fills in the from train computed values into testset
    
    dumps the resulting data into file
    '''    

    # Mean calculations of 'logDemand' for several combinations of Id's
    comb = ['logDemand', 'ProductId', 'ClientId', 'RouteId', 'DepotId']
    week = train.WeekNum.max()
    for j in xrange(len(comb)-1):
        train, test = mean_calc(train, test, col = comb[:j+2])
        train, test = median_week(train, test, col = comb[:j+2], week = week)
    # cope with NAs
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    # Mean and count calculations for the other features
    comb = comb + ['Town', 'State', 'ClientName', 'Pieces', 'Brand', 'Weight', 'ShortName']
    comb.remove('logDemand')
    totalMedian = train['logDemand'].median()
    totalMean = train['logDemand'].mean()
    measures = ['mean', 'count', 'median']
    put = [totalMean, 0 ,totalMedian]
    for col in comb:
        train, test = mult_calc(train, test, col, put = put, measures = measures)
    
    # delete features not both in train and test set
    train, test = drop_feat(train, test)
    
    # dump new data into.csv-files

    train.to_csv(path + '/Data/'+name+'_train.csv', index=False)
    test.to_csv(path + '/Data/'+name+'_test.csv', index = False)
    
    return train, test
    
def build_xgb_feat(train, test, weeks = [3, 4, 5]):
    '''
    builds the features dependent on a split
    '''
    
    train_feat, xgb_feat = split(train, weeks)
    xgb_feat, test = drop_feat(xgb_feat, test)
    train_feat, xgb_feat = build_feat(train_feat, xgb_feat, name = 'xgb_feat')
    
    return train_feat, xgb_feat
    