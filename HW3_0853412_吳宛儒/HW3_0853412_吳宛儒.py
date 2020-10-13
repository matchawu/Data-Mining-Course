# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:08:06 2019

@author: wwj
"""

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import classification_report

#%%
def load_data(dataset):
    if dataset == 'train':
        data = pd.read_csv('training_label.txt',sep='\t',header=None)
    elif dataset == 'test':
        data = pd.read_csv('testing_label.txt',sep='\t',header=None)
    data.columns = ['text']
    data['label'] = data['text'].str[0]
    data['data'] = data['text'].str[10:]
    del data['text']
    return data

#%%
# load data
train = load_data('train')
train = train[:][:10000]
test = load_data('test')

def split_data(dataset):
    X, y = [], []
    X = dataset['data'].values
    y = dataset['label'].values
    return X, y

trainX, trainy, testX, testy = [], [], [], []
trainX, trainy = split_data(train)
testX, testy = split_data(test)


stop_words = set(stopwords.words('english'))

for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.add(w)
    
#%%
# 這個部分 tf-idf 的 transform 有包含，所以可以不用做這個 block
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(trainX)
cool_sw = X.toarray()

feature_sw = cv.get_feature_names()

#%%

# tf-idf
vectorizer = TfidfVectorizer(stop_words='english')
ldf_train = vectorizer.fit_transform(trainX)
ldf_test=vectorizer.transform(testX)
tfidf_feature = vectorizer.get_feature_names()
print(ldf_train.shape)

# ada
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(ldf_train, trainy)
pred_ada = ada.predict(ldf_test)

# xgbc
xgbc = xgb.XGBClassifier()
xgbc.fit(ldf_train,trainy)
xgbc.score(ldf_test, testy)
pred_xgbc = xgbc.predict(ldf_test)

#%%

print('Adaboost:',classification_report(testy, pred_ada))
print('XGboost:',classification_report(testy, pred_xgbc))