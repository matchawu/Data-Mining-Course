# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:51:22 2019

@author: wwj
"""

import pandas as pd
import numpy as np

#%%
'''
0. 讀取資料
'''
#%%

data = pd.read_excel("107年新竹站_20190315.xls")

#%%
'''
1. 資料前處理
'''
#%%
# a. 取出10.11.12月資料
data['month'] = data['日期'].str.split("/", n = 2, expand = True)[1] 
data = data[(data['month'].astype(int) > 9) & (data['month'].astype(int) < 13) ]
data = data.reset_index(drop =  True)

#%%
# b. 缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)
trans = data.drop(['日期','測站','測項','month'], axis = 1)
for i in range(len(trans)):
    for j in range(24):
        k = trans.iloc[i,j]
        if type(k) == str:
            if k != 'NR':
                trans.iloc[i,j] = k[:-1]
                
#%%
yy = pd.DataFrame()
for idx in range(len(trans)):
    #print(idx)
    k = idx+1
    if k%18==0:
        #print(trans[k-18:k])
        temp = trans[k-18:k]
        temp = temp.reset_index(drop=True)
        
        yy = pd.concat([yy,temp],axis=1)

yy = yy.fillna('NaN')
yy.columns = range(yy.shape[1])

#%%
for feature in range(yy.shape[0]):
    for hour in range(yy.shape[1]):
        value = yy.iloc[feature,hour]
        if type(value) == str and value != 'NR':
            preTimes = 1
            posTimes = 1
            pre = yy.iloc[feature,hour-preTimes]
            pos = yy.iloc[feature,hour+posTimes]
            
            while type(pre) == str or pre == 'nan':
                preTimes+=1
                pre = yy.iloc[feature,hour-preTimes]
            while type(pos) == str or pos == 'nan':
                posTimes+=1
                pos = yy.iloc[feature,hour+posTimes]
            
            yy.iloc[feature,hour] = (float(pre)+float(pos))/2
            #print(feature,hour,pre,pos)

#%%
zz = pd.DataFrame()
for idx in range(yy.shape[1]):
    idx+=1
    if idx%24==0:
        temp = yy.iloc[:,idx-24:idx]
        temp.columns = range(temp.shape[1])
        zz = pd.concat([zz,temp], axis=0)
zz = zz.reset_index(drop=True)       

#%%
data = data[['日期','測站','測項','month']]
data = pd.concat([data,zz],axis = 1)
#%%
# c. NR表示無降雨，以0取代 
data = data.where(data != 'NR', 0)

#%%
# d. 將資料切割成訓練集(10.11月)以及測試集(12月) 
train = data.where(data['month'].astype(int) <= 11).dropna(how='all')
test = data.where(data['month'].astype(int) == 12).dropna(how='all')
#%%
# e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料 
# **hint: 將訓練集每18行合併，轉換成維度為(18,61*24)的DataFrame
# (每個屬性都有61天*24小時共1464筆資料) 
train = train.drop(['日期','測站','測項','month'],axis = 1)
test = test.drop(['日期','測站','測項','month'],axis = 1)


def makeTimeSeries(df):
    final = pd.DataFrame()
    for i in range(0,len(df)):
        k = i+1
        if k%18 == 0:
            temp = df.iloc[k-18:k]
            
            temp = temp.reset_index(drop=True)
            #print(temp)
            final = pd.concat([final,temp],axis = 1)
            
    return final

train = makeTimeSeries(train)
test = makeTimeSeries(test)

#%%
'''
2. 時間序列 
'''
#%%
# a. 取6小時為一單位切割，
# 例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時的PM2.5值(Y[0])，
# 下一筆資料為第1~6小時的資料(X[1])去預測第7 小時的PM2.5值(Y[1])  
# *hint: 切割後X的長度應為1464-6=1458 

def buildTrainandTest(df, pastHour=6, futureHour=1):
    X, Y = [], []
    for i in range(df.shape[1]-pastHour):
        X.append(np.array(df.iloc[:,i:i+pastHour]))
        Y.append(np.array(df.iloc[9,i+pastHour:i+pastHour+futureHour]))
    return np.array(X), np.array(Y)

train_X, train_Y = buildTrainandTest(train) 
test_X, test_Y = buildTrainandTest(test) 

#%%
# b. X請分別取 
#  1. 只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值) 
#  2. 所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值) 

def buildTrainandTestPM(df, pastHour=6, futureHour=1):
    X, Y = [], []
    for i in range(df.shape[1]-pastHour):
        X.append(np.array(df.iloc[9,i:i+pastHour]))
        Y.append(np.array(df.iloc[9,i+pastHour:i+pastHour+futureHour]))
    return np.array(X), np.array(Y)

trainPM_X, trainPM_Y = buildTrainandTestPM(train) 
testPM_X, testPM_Y = buildTrainandTestPM(test) 


#%%
# c. 使用兩種模型 Linear Regression 和 Random Forest Regression 建模 
# d. 用測試集資料計算MAE (會有4個結果，2種模型*2種X資料) 
'''
train_X, test_X 3d to 2d
'''

#fix
nsamples, nx, ny = train_X.shape
d2_train_X = train_X.reshape((nsamples,nx*ny))

nsamples, nx, ny = test_X.shape
d2_test_X = test_X.reshape((nsamples,nx*ny))

'''
linear regression
'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

LM_ALL = LinearRegression().fit(d2_train_X, train_Y)
pred_LM_ALL = LM_ALL.predict(d2_test_X)
mae_LM_ALL = mean_absolute_error(test_Y, pred_LM_ALL)

LM_PM = LinearRegression().fit(trainPM_X, trainPM_Y)
pred_LM_PM = LM_PM.predict(testPM_X)
mae_LM_PM = mean_absolute_error(testPM_Y, pred_LM_PM)


'''
random forest regression
'''
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)

RFR_ALL = RFR.fit(d2_train_X, train_Y)
pred_RFR_ALL = RFR_ALL.predict(d2_test_X)
mae_RFR_ALL = mean_absolute_error(test_Y, pred_RFR_ALL)

RFR_PM = RFR.fit(trainPM_X, trainPM_Y)
pred_RFR_PM = RFR_PM.predict(testPM_X)
mae_RFR_PM = mean_absolute_error(testPM_Y, pred_RFR_PM)

'''
print result
'''
print("LinearRegression for all features - MAE:", mae_LM_ALL)
print("LinearRegression for one feature(PM2.5) - MAE:", mae_LM_PM)
print("RandomForestRegressor for all features - MAE:", mae_RFR_ALL)
print("RandomForestRegressor for one feature(PM2.5) - MAE:", mae_RFR_PM)

