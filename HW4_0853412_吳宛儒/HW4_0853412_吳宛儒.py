# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:08:06 2019

@author: wwj
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#RNN & LSTM
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM

import matplotlib.pyplot as plt

#%%
def Plot(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left') 
    plt.show()
    # summarize history for loss 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left') 
    plt.show()

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
train = train[:][:8000]
test = load_data('test')

def split_data(dataset):
    X, y = [], []
    X = dataset['data'].values
    y = dataset['label'].values
    return X, y

trainX, trainy, testX, testy = [], [], [], []
trainX, trainy = split_data(train)
testX, testy = split_data(test)

#%%

# tf-idf
max_features = 5000
vectorizer = TfidfVectorizer(max_features=max_features,stop_words='english')
ldf_train = vectorizer.fit_transform(trainX)
ldf_test = vectorizer.transform(testX)
tfidf_feature = vectorizer.get_feature_names()
print(ldf_train.shape)

#%%
# RNN
modelRNN = Sequential()
modelRNN.add(Embedding(output_dim=128, 
                       input_dim=5000,
                       input_length=5000))
#modelRNN.add(Dropout(0.7)) 
modelRNN.add(SimpleRNN(units=16))
modelRNN.add(Dense(units=128,activation='relu')) 
#modelRNN.add(Dropout(0.35))
modelRNN.add(Dense(units=1,activation='sigmoid'))
modelRNN.summary()
modelRNN.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 
# fit
h_RNN = modelRNN.fit(ldf_train,trainy, 
         epochs=7, 
         batch_size=32,
         verbose=1,
         validation_split=0.2)
#%%
# evaluate
scores = modelRNN.evaluate(ldf_test, testy,verbose=1)
scores[1]

#%%
#predict
predRNN = modelRNN.predict(ldf_test)
predRNN_class = modelRNN.predict_classes(ldf_test)
print(predRNN)
print(predRNN_class)

#%%
# plot
Plot(h_RNN)

#%%
# LSTM

modelLSTM = Sequential()
modelLSTM.add(Embedding(output_dim=64,
                         input_dim=5000,
                         input_length=5000))
modelLSTM.add(Dropout(0.7))
modelLSTM.add(LSTM(32)) 
modelLSTM.add(Dense(units=32,activation='relu')) 
modelLSTM.add(Dropout(0.5))
modelLSTM.add(Dense(units=1,activation='sigmoid'))
modelLSTM.summary()
modelLSTM.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 
# fit
h_LSTM = modelLSTM.fit(ldf_train,trainy, 
         epochs=7, 
         batch_size=128,
         verbose=1,
         validation_split=0.2)

#%%
# evaluate
scores = modelLSTM.evaluate(ldf_test, testy,verbose=1)
scores[1]

#%%
#predict
predLSTM = modelLSTM.predict(ldf_test)
predLSTM_class = modelLSTM.predict_classes(ldf_test)
print(predLSTM)
print(predLSTM_class)

#%%
# plot
Plot(h_LSTM)