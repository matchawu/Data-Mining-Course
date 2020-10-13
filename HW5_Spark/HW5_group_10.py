
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

#we preprocess new csv to drop out the header in cmd and format to .tsv

#loading noheader_data
raw_data = sc.textFile("train_noheader.tsv")
#check first line
raw_data.first()
# In[89]:
#',' to split data
lines = raw_data.map(lambda x: x.split(","))
#count data number
lines.count()

# In[90]:
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import DecisionTreeClassifier
#collect data
data = lines.collect()
#count data
numColumns = len(data[0])

# In[92]:
#define data1，saving the data after processing and format is [(label_1, features_1), (label_2, features_2),…]
data1 = []
for i in range(lines.count()):
    print(i)
    trimmed = [ each.replace('"', " ") for each in data[i] ]
    label = float(trimmed [-1])
    features = map(lambda x: 0.0 if x == "?" else x, trimmed [2:numColumns-1])
    c = (label, Vectors.dense(map(float, features)))
    data1.append(c)

# In[93]:
# to dataframe
df= spark.createDataFrame(data1, ["label","features"])
df.show(10)
# show df 的Schema
df.printSchema()
root
#save df to cache() 
df.cache()

# In[96]:
from pyspark.ml.feature import VectorIndexer
#bulid featureIndexer
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=24).fit(df)
#random split to 75% for training data and 25% for testing data
(trainingData, testData) = df.randomSplit([0.75, 0.25], seed=1234L)

# In[110]:
#buling model
dt = DecisionTreeClassifier(maxDepth=5, labelCol="label", featuresCol="indexedFeatures", impurity="entropy")

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])
#training
model = pipeline.fit(trainingData)      

# In[111]:
#testing 
predictedResultAll = model.transform(testData)
predictedResultAll.select("prediction").show()


# In[120]:
#prediction
df_prediction = predictedResultAll.select("prediction").toPandas()
dtPredictions = list(df_prediction.prediction)
len(dtPredictions)

# In[125]:
#drawing the result
testRaw = testData.count()
testLabel = testData.select("label").collect()


from sklearn.metrics import classification_report
import numpy as np
print(classification_report(np.asarray(testLabel), dtPredictions))


dtTotalCorrect = 0
for i in range(testRaw):
    if dtPredictions[i] == testLabel[i]:
        dtTotalCorrect += 1
print('Accuracy:',1.0 * dtTotalCorrect / testRaw)

