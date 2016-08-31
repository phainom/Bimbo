# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 11:55:15 2016

by Matthäus Deutsch
Computer Science HU Berlin
Project: "Kaggle Bimbo Challenge"


Modell 2: xgboost approach.
"""
#import os
#
#mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
#
#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb
from features2 import split
from matplotlib import pyplot as plt
import pandas as pd
from time import strftime
from numpy import expm1

path = '~/kaggle/Bimbo'

def boosting_modell(train, target, test, evalsplit = [6, 7]):
    '''
    modell computation based on xgboost
    
    Input:
    train - data to train the xgb model
    target - target vector containing the labels for the training set, including WeekNum
    test data for prediction - should contain full features!
    evalsplit
    
    Output: 
    prediction - prediction vector
    '''
    
    # split into training and evaluation (50-50 weeks)
    X_comp, X_eval = split(train, weeks = evalsplit)
    y_comp, y_eval = split(test, weeks = evalsplit) # da stimmt was nicht!
    y_comp = y_comp['logDemand']
    y_eval = y_eval['logDemand']    
    
    # set xgb parameters    
    param = {'objective':'reg:linear', 'booster':'gbtree', 'seed':1234, 'eval_metric':'rmse',
             'nthread':15, 'silent':1, 'eta':0.05, 'gamma':1, 'lambda':2, 'colsample_bylevel':1,
             'colsample_bytree':0.5, 'subsample':1, 'min_child_weight':8, 'max_depth':14}
    
    # compute xgb matrices
    dcomp = xgb.DMatrix(X_comp, y_comp)
    deval = xgb.DMatrix(X_eval, y_eval)
    dtest = xgb.DMatrix(test)
    
    # model specification    
    num_round = 800
    watch = [(dcomp, 'comp'), (deval, 'eval')]
    gbm = xgb.train(param, dcomp, num_round, evals=watch, early_stopping_rounds=20, verbose_eval=True)
    
    # model evaluation
    gbm.dump_model("dump_stats_" + strftime("%Y_%m_%d_%H_%M_%S") + ".txt", with_stats=True)
    gain = pd.Series(gbm.get_score(importance_type="gain"))
    gain = gain.reset_index()
    gain.columns = ["features", "gain"]
    gain.sort_values(by="gain", inplace=True)
    featplot = gain.plot(kind="barh", x="features", y="gain", legend=False, figsize=(10,15))
    plt.title("XGBoost Mean Gain")
    plt.xlabel("Gain")
    fig_featplot = featplot.get_figure()
    fig_featplot.savefig("XGBOOST_GAIN_" + strftime("%Y_%m_%d_%H_%M_%S") + ".png",
                         bbox_inches="tight", pad_inches=1)
    
    print 'xgb done...'
    
    # prediction
    y_pred = gbm.predict(dtest, ntree_limit=gbm.best_ntree_limit)
    submission = pd.DataFrame({"id": id, "Demanda_uni_equil": expm1(y_pred)})
    cols = submission.columns.tolist()
    cols = cols[1:] + cols[0:1]
    submission = submission[cols]
    submission.to_csv("submission_modell2.csv", index=False)

    print 'READY TO GO!'
    

train = pd.read_csv(path + '/Data/xgb_feat_test.csv')
target = pd.read_csv(path + '/Data/target_full.csv')
test = pd.read_csv(path + '/Data/feat_full_test.csv')

target_dump, target_train = split(target)
boosting_modell(train, target_train, test)
