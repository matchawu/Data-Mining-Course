# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:20:09 2019

@author: wwj
"""

#%%
'''
引入套件
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
'''
讀取資料
'''
data = pd.read_csv("train.csv")

#%%
'''
空值補0
'''
data = data.fillna("0")

#%%
'''
PdDistrict和DayofWeek轉成dummies
'''
# dummies
# PdDistrict
PdDistrict = pd.get_dummies(data['PdDistrict'])
print(list(PdDistrict))

# DayofWeek
DayOfWeek = pd.get_dummies(data['DayOfWeek'])
print(list(DayOfWeek))

#%%
'''
切出Hour欄位，並做dummies
'''
# Dates 切開
new = data['Dates'].str.split(" ", n = 1, expand = True) 
data['Date'] = new[0]
data['Time'] = new[1]
del data['Dates']

# 建立新欄位 Hour
data['Hour'] = data['Time'].str.split(":", n = 2, expand = True)[0]

# Hour dummies
Hour = pd.get_dummies(data['Hour'])
print(list(Hour))

#%%
'''
將整理好的dummies合併回原來的data中
'''
# concat to original one
data = pd.concat([data, PdDistrict, DayOfWeek, Hour], axis = 1)
print(list(data))

#%%
'''
取基本模型要用的features
'''
features = ['X', 'Y', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 
            'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday', '00', '01', '02', '03', '04', '05', '06', '07',
            '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'] 

#%%
'''
分開X, Y 
'''
X = data[features]
Y = data['Category']

#%%
'''
亂數切出TRAIN TEST 75%:25%
'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#%%
'''
決策數預測模型
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5,random_state=0)
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

#%%
'''
列出模型評價
'''
print(metrics.accuracy_score(Y_test, Y_pred))
print (metrics.classification_report(Y_test, Y_pred))
confusion_matrix = metrics.confusion_matrix( Y_test, Y_pred, labels=None, sample_weight=None)

#%%
'''
畫出樹
'''
from sklearn import tree
tree.plot_tree(clf.fit(X_train,Y_train)) 

#%%
'''
生成graph檔案
'''
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)
graph.render("result", view=True)

#%%
'''
生成graph
'''
feature_names = list(X_train)
class_names = Y_train.unique()

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=feature_names,  
                      class_names=class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
#print(graph)


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
觀察資料
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 印出欄位
print(list(data))

#%%
# Distribution of crimes
from sklearn.preprocessing import LabelEncoder
le_cat = LabelEncoder()
le_cat.fit(data.Category)
Ytrain = le_cat.transform(data.Category)

plt.hist(Ytrain, bins = 50)

#%%
# The most popular crimes
top_crimes = data.Category.value_counts()[:10]
plt.figure(figsize=(12, 8))
pos = np.arange(len(top_crimes))
plt.barh(pos, top_crimes.values)
plt.yticks(pos, top_crimes.index)

#%%
'''
Address and Category 關係
最常有犯罪的地點
'''
# Address 獨特值的數量
print(len(data.Address.unique()))

# The most criminal locations
top_addresses = data.Address.value_counts()[:20]
plt.figure(figsize=(12, 8))

pos = np.arange(len(top_addresses))
plt.barh(pos, top_addresses.values)
plt.yticks(pos, top_addresses.index)

# Heatmap
subset = data[data.Address.isin(top_addresses.index) & data.Category.isin(top_crimes.index)]
addr_cross_cat = pd.crosstab(subset.Address, subset.Category)

plt.figure(figsize=(10, 10))
sns.heatmap(addr_cross_cat, linewidths=.5)

'''
因為 800 Block of BRYANT ST特別多
因此新增一欄位most分出：800 Block of BRYANT ST 和其他
'''
data['most'] = 0
data.loc[data['Address'] == top_addresses.index[0], 'most'] = 1

#%%
'''
整理欄位
'''
data['Hour'] = data['Time'].str.split(":", n = 2, expand = True)[0].astype(int)
data['Minute'] = data['Time'].str.split(":", n = 2, expand = True)[1].astype(int)
data['Second'] = data['Time'].str.split(":", n = 2, expand = True)[2].astype(int)

#%%
'''
觀察：小時與分鐘欄位值的分布
'''
# 看小時與分鐘是否有特殊分布
data.plot.scatter(x='Hour',y='Minute', color='DarkBlue', alpha=0.1)

#%%
'''
Category and Hour 關係
'''
# Heatmap: Category and Hour
hour_cross_cat = pd.crosstab(data.Category, data.Hour).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(hour_cross_cat, linewidths=.9, cmap="YlGnBu")

#%%
'''
Minute 各值分布
'''
# 單獨看分鐘是否有特殊分布：0和30分鐘分布較多
plt.hist(data['Minute'])

#%%
'''
Category and Minute 關係
'''
minute_cross_cat = pd.crosstab(data.Category, data.Minute).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(minute_cross_cat, linewidths=.9, cmap="Greens")

#%%
'''
新欄位：min_cat
將Minute拆為兩種：0和30分、非0或30分
'''
data['min_cat'] = 0
mask1 = (data['Minute'] == 0)|(data['Minute'] == 30)
data.loc[mask1, 'min_cat'] = 1

#%%
'''
整理欄位
'''
data['Year'] = data['Date'].str.split("-", n=2, expand = True)[0]
data['Month'] =  data['Date'].str.split("-", n = 2, expand = True)[1]
data['Day'] =  data['Date'].str.split("-", n = 2, expand = True)[2]

#%%
'''
Category and Year 關係
'''
# Heatmap: Category and Year
year_cross_cat = pd.crosstab(data.Category, data.Year).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(year_cross_cat, linewidths=.9, cmap="Blues")

#%%
'''
新建欄位Year_norm：
將Year(範圍2003~ 2015)壓到-6 ~ +6之間
'''
data['Year_norm'] = data['Year'].astype(int)-2009

#%%
'''
Category and Month 關係
'''
# Heatmap: Category and Month
month_cross_cat = pd.crosstab(data.Category, data.Month).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(month_cross_cat, linewidths=.9, cmap="Greens")


#%%
#'''
#新建欄位Season：
#'''
## Season
#data['Season'] = data['Date'].str.split("-", n = 2, expand = True)[1]
#
#mask1 = (data['Season'] < '04')
#data.loc[mask1, 'Season'] = 'Spring'
#mask2 = (data['Season'] >= '04')&(data['Season'] <= '06')
#data.loc[mask2, 'Season'] = 'Summer'
#mask3 = (data['Season'] >= '07')&(data['Season'] <= '09')
#data.loc[mask3, 'Season'] = 'Fall'
#mask4 = (data['Season'] >= '10')&(data['Season'] <= '12')
#data.loc[mask4, 'Season'] = 'Winter'
#
#Season = pd.get_dummies(data['Season'])
#data = pd.concat([data, Season], axis = 1)

#%%
#'''
#新建欄位updown_year：
#'''
## 上下半年
## 分上下半年
#data['updown_year'] = '0'
#mask1 = (data['Month'] >= '01')&(data['Month'] <= '06')
#data.loc[mask1, 'updown_year'] = 'up_year'
#mask2 = (data['Month'] >= '07')&(data['Month'] <= '12')
#data.loc[mask2, 'updown_year'] = 'down_year'
#
#updown_year = pd.get_dummies(data['updown_year'])
#data = pd.concat([data, updown_year], axis = 1)


#%%
'''
新建欄位FEB：
'''
# 分是否為二月
data['FEB'] = 0
mask1 = (data['Month'] == '02')
data.loc[mask1, 'FEB'] = 1

#%%
'''
Category and Day 關係
'''
# Heatmap: Category and Day
day_cross_cat = pd.crosstab(data.Category, data.Day).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(day_cross_cat, linewidths=.9, cmap="PiYG")

#%%
'''
Category and DayOfWeek 關係
'''
# Heatmap: Category and DayOfWeek
import seaborn as sns
day_cross_cat = pd.crosstab(data.Category, data.DayOfWeek).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(day_cross_cat, linewidths=.9, cmap="BuPu")

#%%
'''
經緯度
'''
# 稍加整理
#data['location'] = data['X'].round(2).astype(str) + " " + data['Y'].round(2).astype(str)
#data['X_round'] = data['X'].round(2).astype(str)
#data['Y_round'] = data['Y'].round(2).astype(str)
#
data.plot.scatter(x='X',y='Y',color='DarkBlue',label='locate')

df = data[['X','Y']]
df = data[data.X != -120.50]
plt.figure(figsize=(100,50))
df.plot.scatter(x='X',y='Y', color='DarkBlue', alpha=0.1)

#%%
'''
Category and Resolution 關係

'''
print(data.Resolution.unique())

# Heatmap: Category and Resolution
resolution_cross_cat = pd.crosstab(data.Category, data.Resolution).apply(lambda r: r/r.sum(), axis=1)

plt.figure(figsize=(10, 10))
sns.heatmap(resolution_cross_cat, linewidths=.9, cmap="PiYG")

'''
大部分集中在ARREST, BOOKED、NONE
做法：分為ARREST, BOOKED、NONE、OTHERS三類去做dummies
'''
# dummies
data['res'] = data['Resolution']
data.loc[(data.Resolution != 'NONE') & (data.Resolution != 'ARREST, BOOKED'), 'res'] = 'other'
print(data.res.unique())
res = pd.get_dummies(data['res'])
print(list(res))

data = pd.concat([data,res],axis=1,ignore_index=False)


#%%
'''
最終選定features
'''
features = ['X', 'Y', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 
            'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday', '00', '01', '02', '03', '04', '05', '06', '07',
            '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 'Year','ARREST, BOOKED', 'NONE',
            'other','most','min_cat'] 

#%%
X = data[features]
Y = data['Category']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#%%
'''
決策數預測模型
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5,random_state=0)
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

#%%
'''
列出模型評價
'''
print(metrics.accuracy_score(Y_test, Y_pred))
print (metrics.classification_report(Y_test, Y_pred))
confusion_matrix = metrics.confusion_matrix( Y_test, Y_pred, labels=None, sample_weight=None)

#%%
'''
畫出樹
'''
from sklearn import tree
tree.plot_tree(clf.fit(X_train,Y_train)) 

#%%
'''
生成graph檔案
'''
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)
graph.render("result", view=True)

#%%
'''
生成graph
'''
feature_names = list(X_train)
class_names = Y_train.unique()

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=feature_names,  
                      class_names=class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  

