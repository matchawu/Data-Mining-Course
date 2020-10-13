
# coding: utf-8

# In[11]:

import pandas as pd

#----processing data---#

#loading data
data = pd.read_csv('character-deaths.csv')
#chosing ['Death Year'] as label
#nan filled in 0 and if death, filling in 1
data['Death Year'][data['Death Year'] >0]= 1
data['Death Year'] = data['Death Year'].fillna(0)
#drop columns ['Book of Death', 'Death Chapter']
data = data.drop(columns= ['Book of Death', 'Death Chapter'])
#processing get_dummies
Allegiances = pd.get_dummies(data.Allegiances)
#concating raw data and dummies
data = pd.concat([data, Allegiances], axis = 1)
#nan filled in mean
data['Book Intro Chapter'] = data['Book Intro Chapter'].fillna(int(data['Book Intro Chapter'].mean()))
#drop columns
data = data.drop(columns= ['Allegiances'])
#let label to last column
label = data['Death Year']
data = data.drop(columns = ['Death Year'])
data = pd.concat([data, label], axis = 1)
#to a new csv
data.to_csv('train_data.csv')


# In[88]:

#we preprocess new csv to drop out the header in cmd and 
#loading raw_data

raw_data = sc.textFile("train_noheader.tsv")
raw_data.first()


# In[89]:

lines = raw_data.map(lambda x: x.split(","))
lines.count()


# In[90]:

# print(str(lines.count()))
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import DecisionTreeClassifier

data = lines.collect()
numColumns = len(data[0])
numColumns


# In[91]:

data[-1]


# In[92]:

data1 = []

for i in range(lines.count()):
    print(i)
    trimmed = [ each.replace('"', " ") for each in data[i] ]
    label = float(trimmed [-1])
    features = map(lambda x: 0.0 if x == "?" else x, trimmed [2:numColumns-1])
    c = (label, Vectors.dense(map(float, features)))
    data1.append(c)


# In[93]:

df= spark.createDataFrame(data1, ["label","features"])
df.show(10)


# In[94]:

df.printSchema()
root


# In[95]:

df.cache()


# In[96]:

from pyspark.ml.feature import VectorIndexer
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=24).fit(df)


# In[109]:

(trainingData, testData) = df.randomSplit([0.75, 0.25],seed=1234L)
trainingData.count()


# In[110]:

dt = DecisionTreeClassifier(maxDepth=5, labelCol="label", featuresCol="indexedFeatures", impurity="entropy")

from pyspark.ml import Pipeline
 
pipeline = Pipeline(stages=[featureIndexer, dt])
model = pipeline.fit(trainingData)      ## 训练模型


# In[105]:

data1[0]


# In[111]:

predictedResultAll = model.transform(testData)
predictedResultAll.select("prediction").show()


# In[120]:

df_prediction = predictedResultAll.select("prediction").toPandas()
dtPredictions = list(df_prediction.prediction)
len(dtPredictions)


# In[125]:

testRaw = testData.count()
testLabel = testData.select("label").collect()


# In[128]:

from sklearn.metrics import classification_report
import numpy as np
print(classification_report(np.asarray(testLabel), dtPredictions))


# In[130]:

dtTotalCorrect = 0
for i in range(testRaw):
    if dtPredictions[i] == testLabel[i]:
        dtTotalCorrect += 1
print('Accuracy:',1.0 * dtTotalCorrect / testRaw)


# In[ ]:



