#!/usr/bin/env python
# coding: utf-8

# In[2]:


# loading package
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.56)
from sklearn.ensemble import RandomForestClassifier # 隨機森林
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV


# In[51]:


# for display dataframe
from IPython.display import display
from IPython.display import display_html


# In[49]:


def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"')
                 ,raw=True)


# In[48]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


# loading data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_data = df_train.append(df_test)


# In[4]:


list(df_data)


# In[ ]:


# Sex - observation


# In[5]:


# 以Sex為x軸 計算各Sex的生死數量
sns.countplot(df_data['Sex'], hue=df_data['Survived'])


# In[6]:


# 計算每種性別(group by sex)的平均生存率df_data['Survived'].mean()
display(df_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().round(3))


# In[ ]:


# 可見:
# 大部分的男性都掛了(僅剩18%存活)，而女性則大部分都存活了下來(~75%)


# In[ ]:


# Pclass 艙等 - observation


# In[8]:


sns.countplot(df_data['Pclass'], hue=df_data['Survived'])


# In[9]:


display(df_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().round(3))


# In[ ]:


# 可見
# 頭等艙生存率較高


# In[ ]:


# 轉換性別資料(英文字變成0,1) 0為女性，1為男性


# In[10]:


# 性別欄位本來的樣子
df_data['Sex']


# In[11]:


# 轉換
df_data['Sex_Code'] = df_data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')


# In[12]:


# 新建的Sex_Code欄位
df_data['Sex_Code']


# In[ ]:


# 切出訓練集、測試集


# In[13]:


df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]

# In[14]:

X = df_train.drop(labels = ['Survived','PassengerId'], axis = 1)
Y = df_train['Survived']

# In[19]:

# 以性別和艙等作為特徵
Base = ['Sex_Code', 'Pclass']
# 建立隨機森林
Base_Model = RandomForestClassifier(random_state = 2, 
                                    n_estimators = 250, 
                                    min_samples_split = 20, 
                                    oob_score = True)
# fit model
Base_Model.fit(X[Base], Y)
print('Base oob score : %.5f' % (Base_Model.oob_score_))

# Fare 票價 - observation


# In[ ]:

sns.countplot(df_data['Fare'], hue=df_data['Survived'])

# In[23]:

display(df_data[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean().round(3))

# In[ ]:

# 取log以後畫出示意圖
# 票價分布非常廣，也很傾斜，log可以解決這項問題

# In[32]:

fig, ax = plt.subplots(figsize = (18,7))
df_data['Log_Fare'] = (df_data['Fare']+1).map(lambda x: np.log10(x) if x > 0 else 0)
sns.boxplot(y='Pclass', x='Log_Fare', 
            hue='Survived', data=df_data, 
            orient='h', ax=ax, palette="Set3")
ax.set_title(' Log_Fare & Pclass vs Survived ', fontsize = 20)
pd.pivot_table(df_data,values = ['Fare'], 
               index = ['Pclass'], 
               columns=['Survived'], 
               aggfunc = 'median').round(3)

# In[36]:

# 先填補缺失值
df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())


# 將票價分別切成4,5,6個區間，命名並做為新的特徵

df_data['FareBin_4'] = pd.qcut(df_data['Fare'], 4)
df_data['FareBin_5'] = pd.qcut(df_data['Fare'], 5)
df_data['FareBin_6'] = pd.qcut(df_data['Fare'], 6)

# label encoder()
label = LabelEncoder()
df_data['FareBin_Code_4'] = label.fit_transform(df_data['FareBin_4'])
df_data['FareBin_Code_5'] = label.fit_transform(df_data['FareBin_5'])
df_data['FareBin_Code_6'] = label.fit_transform(df_data['FareBin_6'])

# cross tab
df_4 = pd.crosstab(df_data['FareBin_Code_4'],df_data['Pclass'])
df_5 = pd.crosstab(df_data['FareBin_Code_5'],df_data['Pclass'])
df_6 = pd.crosstab(df_data['FareBin_Code_6'],df_data['Pclass'])

display_side_by_side(df_4,df_5,df_6)


# In[56]:

# plots
fig, [ax1,ax2,ax3] = plt.subplots(1,3,sharey= True)
fig.set_figwidth(18)
for axi in [ax1,ax2,ax3]:
    axi.axhline(0.5,linestyle='dashed',c='black',alpha=.3)

g1 = sns.factorplot(x='FareBin_Code_4',y="Survived",data=df_data,kind='bar',ax=ax1)
g2 = sns.factorplot(x='FareBin_Code_5',y="Survived",data=df_data,kind='bar',ax=ax2)
g3 = sns.factorplot(x='FareBin_Code_6',y="Survived",data=df_data,kind='bar',ax=ax3)

# close FacetGrid object
plt.close(g1.fig)
plt.close(g2.fig)
plt.close(g3.fig)


# In[ ]:

df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]

X = df_train.drop(labels=['Survived', 'PassengerId'], axis=1)
Y = df_train['Survived']

# In[65]:

compare = ['Sex_Code','Pclass','FareBin_Code_4','FareBin_Code_5','FareBin_Code_6']

# In[66]:

X['FareBin_Code_4']

# In[67]:

selector = RFECV(RandomForestClassifier(n_estimators=250,min_samples_split=20),cv=10,n_jobs=-1)
selector.fit(X[compare],Y)


# In[69]:

print(selector.support_)
print(selector.ranking_)
print(selector.grid_scores_*100)


# In[70]:

score_b4, score_b5, score_b6 = [],[],[]
seeds=10
for i in range(seeds):
    diff_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    selector = RFECV(RandomForestClassifier(random_state=i,
                                           n_estimators=250,
                                           min_samples_split=20),
                    cv=diff_cv, n_jobs=-1)
    selector.fit(X[compare],Y)
    score_b4.append(selector.grid_scores_[2])
    score_b5.append(selector.grid_scores_[3])
    score_b6.append(selector.grid_scores_[4])


# In[73]:

score_list = [score_b4, score_b5, score_b6]
for item in score_list:
    item = np.array(item*100)
#plot
fig = plt.figure(figsize=(18,8))
ax = plt.gca()
ax.plot(range(seeds), score_b4, '-ok', label='bins = 4')
ax.plot(range(seeds), score_b5, '-og', label='bins = 5')
ax.plot(range(seeds), score_b6, '-ob', label='bins = 6')
ax.set_xlabel("Seed #", fontsize= '14')
ax.set_ylim(0.783,0.815)
ax.set_ylabel("Accuracy", fontsize = '14')
ax.set_title('bins = 4 vs bins = 5 vs bins = 6', fontsize = '20')
plt.legend(fontsize = 14, loc='upper right')

# In[75]:

b4,b5,b6 = ['Sex_Code','Pclass','FareBin_Code_4'],['Sex_Code','Pclass','FareBin_Code_5'],['Sex_Code','Pclass','FareBin_Code_6']

# In[81]:

b4_Model = RandomForestClassifier(random_state=2,
                                 n_estimators=250,
                                 min_samples_split=20,
                                 oob_score=True)
b4_Model.fit(X[b4], Y)


# In[82]:

b5_Model = RandomForestClassifier(random_state=2,
                                 n_estimators=250,
                                 min_samples_split=20,
                                 oob_score=True)
b5_Model.fit(X[b5], Y)


# In[83]:

b6_Model = RandomForestClassifier(random_state=2,
                                 n_estimators=250,
                                 min_samples_split=20,
                                 oob_score=True)
b6_Model.fit(X[b6], Y)


# In[84]:


print('b4 oob score :%.5f' %(b4_Model.oob_score_),
      '  LB_PUBLIC : 0.7790')
print('b5 oob score :%.5f' %(b5_Model.oob_score_),
      '  LB_PUBLIC : 0.79425')
print('b6 oob score :%.5f' %(b6_Model.oob_score_),
      '  LB_PUBLIC : 0.77033')


# In[85]:

# 將b5_Model提交至Kaggle

X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

b5_pred = b5_Model.predict(X_Submit[b5])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":b5_pred.astype(int)})
submit.to_csv("submit_b5.csv", index=False)


# In[90]:

# Ticket - observation

# 發現了乘客持有相同的船票意味著他們可能是家人或是朋友，
# 而在訓練集上這些互相有連結的人常常是一起活下來或是一起喪命

df_train['Ticket'].describe()


# In[104]:

df_data['Family_size'] = df_data['SibSp'] + df_data['Parch'] + 1

# In[108]:

deplicate_ticket = []
for tk in df_data.Ticket.unique():
    tem = df_data.loc[df_data.Ticket == tk, 'Fare']
    #print(tem.count())
    if tem.count() > 1:
        deplicate_ticket.append(df_data.loc[df_data.Ticket == tk,
                                            ['Name','Ticket',
                                             'Fare','Cabin',
                                             'Family_size',
                                             'Survived']])
# 具有相同票根 (dataframe)
deplicate_ticket = pd.concat(deplicate_ticket)

# 列出前14筆具有相同票根的人類
deplicate_ticket.head(14)


# In[109]:


df_fri = deplicate_ticket.loc[(deplicate_ticket.Family_size == 1) & (deplicate_ticket.Survived.notnull())].head(7)
df_fami = deplicate_ticket.loc[(deplicate_ticket.Family_size > 1) & (deplicate_ticket.Survived.notnull())].head(7)


# In[111]:


display(df_fri,df_fami)


# In[112]:


print('people keep the same ticket: %.0f' %len(deplicate_ticket))
print('friends: %.0f'%len(deplicate_ticket[deplicate_ticket.Family_size == 1]))
print('familes: %.0f'%len(deplicate_ticket[deplicate_ticket.Family_size > 1]))


# In[113]:

# 沒有生還資訊者Connected_Survival設為0.5
df_data['Connected_Survival'] = 0.5

for _, df_grp in df_data.groupby('Ticket'):
    if(len(df_grp) > 1): # 有某種票根的人數大於一人
        for ind, row in df_grp.iterrows():
            
            smax = df_grp.drop(ind)['Survived'].max()
            smin = df_grp.drop(ind)['Survived'].min()

            passID = row['PassengerId']
            if(smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 
                            'Connected_Survival'] = 1
                #如果群組中有人生還 則定義 Connected_Survival = 1
            elif(smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 
                            'Connected_Survival'] = 0
                #沒有人生還，則定義Connected_Survival = 0 

print('people keep the same ticket: %.0f'%len(deplicate_ticket))
print('people have connected information: %.0f'%(df_data[df_data['Connected_Survival'] != 0.5].shape[0]))
df_data.groupby('Connected_Survival')[['Survived']].mean().round(3)


# In[120]:

df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]


# In[123]:

X = df_train.drop(labels = ['Survived','PassengerId'], axis = 1)
Y = df_train['Survived']


# In[124]:

connect = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival']
connect_Model = RandomForestClassifier(random_state=2, 
                                       n_estimators=250, 
                                       min_samples_split=20,
                                       oob_score=True)
connect_Model.fit(X[connect], Y)
print('connect oob score :%.5f'%(connect_Model.oob_score_))


# In[125]:

# 提交至kaggle

X_Submit = df_test.drop(labels=['PassengerId'], axis = 1)

connect_pred = connect_Model.predict(X_Submit[connect])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived": connect_pred.astype(int)})
submit.to_csv("submit_connect.csv",index=False)


# In[129]:


# Age - observation

df_data['Has_Age'] = df_data['Age'].isnull().map(lambda x : 0 if x == True else 1)

fig, [ax1, ax2] = plt.subplots(1,2)
fig.set_figwidth(18)

ax1 = sns.countplot(df_data['Pclass'], hue = df_data['Has_Age'], ax = ax1)
ax2 = sns.countplot(df_data['Sex'], hue = df_data['Has_Age'], ax = ax2)

pd.crosstab(df_data['Has_Age'], df_data['Sex'], margins=True).round(3)


# In[137]:


Mask_Has_Age_P12_Survived = ((df_data.Has_Age == 1)&(df_data.Pclass != 3)&(df_data.Survived == 1))
Mask_Has_Age_P12_Dead = ((df_data.Has_Age == 1)&(df_data.Pclass != 3)&(df_data.Survived == 0))


# In[139]:

fig, ax = plt.subplots( figsize = (15,9))
ax = sns.distplot(df_data.loc[Mask_Has_Age_P12_Survived, 'Age'], 
                  kde=False, bins=10, norm_hist=True,
                  label='Survived')
ax = sns.distplot(df_data.loc[Mask_Has_Age_P12_Dead, 'Age'],
                  kde=False, bins=10, norm_hist=True,
                  label='Dead')
ax.legend()
ax.set_title('Age vs Survived in Pclass = 1 and 2', fontsize = 20)


# In[ ]:

df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_data['Title'] = df_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
df_data['Title'] = df_data['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2,"Miss" : 3, "Mrs" : 4 })
Ti = df_data.groupby('Title')['Age'].median()
Ti


# In[150]:

Ti_pred = df_data.groupby('Title')['Age'].median().values
df_data['Ti_Age'] = df_data['Age']
# Filling the missing age
for i in range(0,5):
 # 0 1 2 3 4 5
    df_data.loc[(df_data.Age.isnull()) & (df_data.Title == i),'Ti_Age'] = Ti_pred[i]
df_data['Ti_Age'] = df_data['Ti_Age'].astype('int')
df_data['Ti_Minor'] = ((df_data['Ti_Age']) < 16.0) * 1


# In[152]:

# splits again beacuse we just engineered new feature
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']


# In[153]:


minor = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival','Ti_Minor']
minor_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
minor_Model.fit(X[minor], Y)
print('minor oob score :%.5f' %(minor_Model.oob_score_))


# In[154]:


# submits
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

minor_pred = minor_Model.predict(X_Submit[minor])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":minor_pred.astype(int)})
submit.to_csv("submit_minor.csv",index=False)


# In[ ]:


#家庭人數(Family_size)


g = sns.factorplot(x='Family_size', y='Survived',data=df_data)
g = g.set_ylabels("Survival Probability")


# In[ ]:

# cut into 3 class
df_data['L_Family'] = df_data['Family_size'].apply(lambda x: 0 if x<= 4 else 1).astype(int)
df_data.loc[ df_data['Family_size'] == 1, 'FamilyClass'] = 0
df_data.loc[ (df_data['Family_size'] <= 4) & (df_data['Family_size'] > 1), 'FamilyClass'] = 1
df_data.loc[ df_data['Family_size'] >= 5, 'FamilyClass'] = 2
df_data['FamilyClass'] = df_data['FamilyClass'].astype(int) 
df_data[['FamilyClass','Survived']].groupby(['FamilyClass']).mean()


# In[ ]:

pd.crosstab(df_data['Family_size'],df_data['Sex']).plot(kind='bar',stacked=True,title="Sex")


# In[ ]:

Minor_mask = (df_data.Ti_Minor == 1)
fig, [ax1, ax2] = plt.subplots(1, 2)
fig.set_figwidth(18)
sns.countplot(df_data[Minor_mask]['Family_size'],ax=ax1)
ax1.set_title("Minor counts")
pd.crosstab(df_data[Minor_mask]['Family_size'],df_data[Minor_mask]['Survived']).plot(kind='bar',stacked=True,title="Survived Minor counts",ax=ax2)


