# -*- coding: utf-8 -*-
"""HotelReview_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N6cFQx9bz411rXVWl_ylbMn1rEYp__rl
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
#%%
hotel = pd.read_csv('Hotel_Reviews.csv', sep = ',')
"""data preprocess"""
train = hotel.copy()
nn=train[train['Negative_Review']=='No Negative']
np=train[train['Positive_Review']=='No Positive']
np.count()
a=nn['Reviewer_Score'].mean()
b=np['Reviewer_Score'].mean()
print('no negative review avg score:',a)
print('no positive review avg score:',b)

train=train[(train['Negative_Review']=='No Negative') | (train['Positive_Review']=='No Positive')]
t=train.loc[:,['Negative_Review','Positive_Review']]
t['sentiment']=0
t['sentiment'][t['Negative_Review']=='No Negative']=1
t[0:10]
neg_rev=t[t['sentiment']==0]
neg_rev=neg_rev.drop(columns=['Positive_Review'])
pos_rev=t[t['sentiment']==1]
pos_rev=pos_rev.drop(columns=['Negative_Review'])
pos_rev['Positive_Review']=pos_rev['Positive_Review'].str.lower()
neg_rev['Negative_Review']=neg_rev['Negative_Review'].str.lower()
pos_rev[0:2]
pos_rev=pos_rev.reset_index(drop=True)
neg_rev=neg_rev.reset_index(drop=True)
pos_rev=pos_rev.rename(index=str, columns={"Positive_Review": "review"})
neg_rev=neg_rev.rename(index=str, columns={"Negative_Review": "review"})
rev=pd.concat([pos_rev,neg_rev],axis=0)
rev2=rev.copy()
aa=rev.groupby('sentiment').count()
#%%
"""one hot"""

rev2['negative']=rev2['sentiment']
rev2['negative'].replace([1,0], [0,1], inplace = True)
rev2=rev2.rename(index=str, columns={"sentiment": "positive"})
"""stop word"""
nltk.download('stopwords')
stop = stopwords.words("english") 
rev2['review'] = rev2['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

review_train, review_test, label_train, label_test = train_test_split(rev2['review'],rev2.loc[:,['positive','negative']], test_size=0.2,random_state=13,stratify=rev['sentiment'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_train)
tokenizer.num_words=2000
X_train = tokenizer.texts_to_sequences(review_train)
X_test = tokenizer.texts_to_sequences(review_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


maxlen = 15

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

X_train = tokenizer.sequences_to_texts(X_train)
X_test = tokenizer.sequences_to_texts(X_test)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)
X_train = tfidf_vectorizer.transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)

X_train=X_train.toarray()
X_test=X_test.toarray()


X_train=X_train.reshape(130967, 1979,1)
X_test=X_test.reshape(32742, 1979,1)


#%%
rev['review'] = rev['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
nltk.download('wordnet')
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

rev['review'] = get_lemmatized_text(rev['review'])

##cnn上無用

def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

rev['review'] = get_stemmed_text(rev['review'])


##cnn上無用
#%%
"""### -split"""

review_train, review_test, label_train, label_test = train_test_split(rev['review'],rev['sentiment'], test_size=0.2,random_state=13,stratify=rev['sentiment'])

#%%
"""### CNN model"""

embedding_dim = 200
model = Sequential()
model.add(layers.Conv1D(256,5 ,activation='relu', input_shape=(1979,1)))

model.add(layers.Conv1D(128, 5,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(64, 5,activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history10 = model.fit(X_train, label_train,
                    validation_split=0.2,
                    epochs=7, #從50一直試到3
                    verbose=2,
                    batch_size=50)

"""### predict"""

loss, accuracy = model.evaluate(X_train, label_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, label_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

test_y_predicted = model.predict(X_test)
test_y_predicted[0:10]

aa=test_y_predicted.copy()
for i in range(len(aa)):
    if aa[i]<0.75:
        aa[i]=0
    else:
        aa[i]=1



print(classification_report(label_test, aa))

history=history10
print(history.history.keys())


print('means loss',np.mean(history.history["loss"]))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()

test_loss,test_acc=model.evaluate(x=X_test,y=label_test,batch_size=10,verbose=1)

#ACC 在分類的問題答案才會一模一樣==>可以算準確率 而在回歸的問題上很難預測到的一模一樣的答案所以沒有再算準確率
print('means accuracy',np.mean(history.history["acc"]))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuary')
plt.ylabel('accuary')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()
#%%
"""### **1-2 logistic regression+token**"""

#SPLIT
review_train2, review_test2, label_train2, label_test2 = train_test_split(rev2['review'],rev2['sentiment'], test_size=0.2,stratify=rev['sentiment'])


#TOKEN
tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_train2)

X_train2 = tokenizer.texts_to_sequences(review_train2)
X_test2 = tokenizer.texts_to_sequences(review_test2)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(review_train2[2])
print(X_train2[2])

#maxlen = 15
X_train2 = pad_sequences(X_train2, padding='post', maxlen=maxlen)
X_test2 = pad_sequences(X_test2, padding='post', maxlen=maxlen)

print(X_train2[0:2])

# Commented out IPython magic to ensure Python compatibility.
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train2, label_train2)
    print ("Accuracy for C=%s: %s" 
            % (c, accuracy_score(label_test2, lr.predict(X_test2))))
    
#有沒有用stop word都一樣

test_y_predicted = lr.predict(X_test2)
print(classification_report(label_test2, test_y_predicted))

"""### 1-3 logistic regression+tfidf"""

review_train3, review_test3, label_train3, label_test3 = train_test_split(rev['review'],rev['sentiment'], test_size=0.2,stratify=rev['sentiment'])

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(review_train3)
X_train3 = tfidf_vectorizer.transform(review_train3)
X_test3 = tfidf_vectorizer.transform(review_test3)

# Commented out IPython magic to ensure Python compatibility.
for c in [0.1,0.25, 0.5, 1,1.5,2]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, label_train)
    print ("Accuracy for C=%s: %s" 
#            % (c, accuracy_score(label_test, lr.predict(X_test))))
#有沒有stop word真的沒差

final_tfidf = LogisticRegression(C=1.5)
final_tfidf.fit(X_train, label_train)
print ("Accuracy:", accuracy_score(label_test, lr.predict(X_test)))
test_y_predicted = final_tfidf.predict(X_test)
print(classification_report(label_test, test_y_predicted))

"""### 1-4.logistic regression + tfidf + ngram"""

# Commented out IPython magic to ensure Python compatibility.
review_train4, review_test4, label_train4, label_test4 = train_test_split(rev['review'],rev['sentiment'], test_size=0.2,stratify=rev['sentiment'])

tfidf_vectorizer4 = TfidfVectorizer(ngram_range=(1, 2))
tfidf_vectorizer4.fit(review_train4)
X_train4 = tfidf_vectorizer4.transform(review_train4)
X_test4 = tfidf_vectorizer4.transform(review_test4)

for c in [0.1,0.25, 0.5, 1,1.5,2]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train4, label_train4)
    print ("Accuracy for C=%s: %s" 
#            % (c, accuracy_score(label_test4, lr.predict(X_test4))))

final_tfidf4 = LogisticRegression(C=2)
final_tfidf4.fit(X_train4, label_train4)
print ("Accuracy:", accuracy_score(label_test4, lr.predict(X_test4)))
test_y_predicted4 = final_tfidf4.predict(X_test4)
print(classification_report(label_test4, test_y_predicted4))

"""### undersampling+logistic regression"""

review_train6, review_test6, label_train6, label_test6 = train_test_split(rev['review'],rev['sentiment'], test_size=0.2,random_state=13,stratify=rev['sentiment'])

train6 = pd.concat( [review_train5,label_train5] , axis=1, sort=False)
train6[0:4]
train6.groupby('sentiment').count()

from sklearn.utils import resample
#1 gender:
s_majority = train6[train6['sentiment'] == 1]
s_minority = train6[train6['sentiment'] == 0]

# Downsample majority class
s_majority_downsampled = resample(s_majority, replace=False,
                                   n_samples=28655, random_state=123)
s_downsampled = pd.concat([s_majority_downsampled, s_minority])
print(s_downsampled['sentiment'].value_counts())

review_train6=s_downsampled['review']
label_train6=s_downsampled['sentiment']

import nltk
nltk.download('wordnet')
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

s_downsampled['review'] = get_lemmatized_text(s_downsampled['review'])

def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

s_downsampled['review'] = get_stemmed_text(s_downsampled['review'])

# Commented out IPython magic to ensure Python compatibility.
tfidf_vectorizer6 = TfidfVectorizer(ngram_range=(1, 2))
tfidf_vectorizer6.fit(review_train6)
X_train6 = tfidf_vectorizer6.transform(review_train6)
X_test6 = tfidf_vectorizer6.transform(review_test6)

for c in [4.5,5,5.5,6,6.5,7]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train6, label_train6)
    print ("Accuracy for C=%s: %s" 
#            % (c, accuracy_score(label_test6, lr.predict(X_test6))))

final_tfidf6 = LogisticRegression(C=6)
final_tfidf6.fit(X_train6, label_train6)
print ("Accuracy:", accuracy_score(label_test6, lr.predict(X_test6)))
test_y_predicted6 = final_tfidf6.predict(X_test6)
print(classification_report(label_test6, test_y_predicted6))

"""### 2.cnn+undersampling"""

review_train5, review_test5, label_train5, label_test5 = train_test_split(rev['review'],rev['sentiment'], test_size=0.2,random_state=13,stratify=rev['sentiment'])

train5 = pd.concat( [review_train5,label_train5] , axis=1, sort=False)
train5[0:4]

train5.groupby('sentiment').count()

from sklearn.utils import resample
#1 gender:
s_majority = train5[train5['sentiment'] == 1]
s_minority = train5[train5['sentiment'] == 0]

# Downsample majority class
s_majority_downsampled = resample(s_majority, replace=False,
                                   n_samples=28655, random_state=123)
s_downsampled = pd.concat([s_majority_downsampled, s_minority])
print(s_downsampled['sentiment'].value_counts())

review_train5=s_downsampled['review']
label_train5=s_downsampled['sentiment']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_train5)
tokenizer.num_words=2000
X_train5 = tokenizer.texts_to_sequences(review_train5)
X_test5 = tokenizer.texts_to_sequences(review_test5)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(review_train5[2])
print(X_train5[2])

vocab_size

maxlen = 15

X_train5 = pad_sequences(X_train5, padding='post', maxlen=maxlen)
X_test5 = pad_sequences(X_test5, padding='post', maxlen=maxlen)

print(X_train5[7])

X_train5[1:5]

X_train5 = tokenizer.sequences_to_texts(X_train5)
X_test5 = tokenizer.sequences_to_texts(X_test5)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train5)
X_train5 = tfidf_vectorizer.transform(X_train5)
X_test5 = tfidf_vectorizer.transform(X_test5)

X_train5=X_train5.toarray()
X_test5=X_test5.toarray()

X_test5.shape

X_train5=X_train5.reshape((57310, 1972,1))
X_test5=X_test5.reshape((32742, 1972,1))

embedding_dim = 200
model5 = Sequential()
model5.add(layers.Conv1D(256,5 ,activation='relu', input_shape=(1972,1)))

model5.add(layers.Conv1D(128, 5,activation='relu'))
model5.add(layers.Dropout(0.2))
model5.add(layers.Conv1D(64, 5,activation='relu'))
model5.add(layers.Dropout(0.2))

model5.add(layers.GlobalMaxPooling1D())
model5.add(layers.Dense(100, activation='relu'))
model5.add(layers.Dense(1, activation='sigmoid'))
model5.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model5.summary()

history5 = model5.fit(X_train5, label_train5,
                    validation_split=0.2,
                    epochs=5,
                    verbose=2,
                    batch_size=50)

loss, accuracy = model5.evaluate(X_train5, label_train5, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model5.evaluate(X_test5, label_test5, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
test_y_predicted5 = model5.predict(X_test5)

bb=test_y_predicted5.copy()
for i in range(len(bb)):
    if bb[i]<0.5:
        bb[i]=0
    else:
        bb[i]=1
print(classification_report(label_test5, bb))

print('means loss',np.mean(history5.history["loss"]))
plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()

test_loss,test_acc=model5.evaluate(x=X_test5,y=label_test5,batch_size=10,verbose=1)

#ACC 在分類的問題答案才會一模一樣==>可以算準確率 而在回歸的問題上很難預測到的一模一樣的答案所以沒有再算準確率
print('means accuracy',np.mean(history5.history["acc"]))
plt.plot(history5.history['acc'])
plt.plot(history5.history['val_acc'])
plt.title('model accuary')
plt.ylabel('accuary')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()

test_y_predicted