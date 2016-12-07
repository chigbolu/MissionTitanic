from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale



#Python version required: 2.7
#https://www.kaggle.com/c/titanic
import csv as csv
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import unicodedata

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation, PredictionOutput
import weka.core.converters as converters
from weka.core.converters import Loader, Saver
from weka.core.classes import Random
import re
from weka.filters import Filter, MultiFilter
from weka.core.classes import OptionHandler, join_options


titlesAverages = dict.fromkeys(['Miss','Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
    'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])


def preprocessing(superfile):

  df = pd.read_csv(superfile, header=0)

  for title in titlesAverages:
    rows = df[df.Name.str.contains(' '+title+'.')]
    average = rows['Age'].mean()

    if math.isnan(average):
      titlesAverages[title] = int(25)
    else:
      titlesAverages[title] = int(average)

  for index,row in df.iterrows():
    age = row['Age']
    if(math.isnan(age)):
      title = getTitle(row['Name'])
      if title in titlesAverages:
        average = titlesAverages[title]
        df.set_value(index, 'Age', average)
      else:
        df.set_value(index, 'Age', 25)

  for index, row in df.iterrows():
    embark = row['Embarked']
    if(pd.isnull(embark)):
      df.set_value(index,'Embarked','C')

  del df['Name']
  del df['Fare']
  del df['Ticket']
  del df['Cabin']
  #if 'Survived' in df:
  return df
  #else:
  #  buckets = [0] * len(df)
  #  df.insert(1,'Survived', buckets)
  #  return df




def getTitle(name):
    for title in titlesAverages:
        if title + '.' in name:
            return title
    return 'None'

jvm.start()

dfOne = preprocessing("train.csv")
temp = pd.DataFrame({'PassengerId':[]})
temp['PassengerId'] = dfOne['PassengerId']
del dfOne['PassengerId']
dfOne['Survived'] = dfOne['Survived'].map({1: 'Y', 0: 'N'})
dfOne.to_csv("trainComplete.csv",index = False)


dfTwo = preprocessing("test.csv")
tempTest = pd.DataFrame({'PassengerId':[]})
tempTest['PassengerId'] = dfTwo['PassengerId']
del dfTwo['PassengerId']
#dfTwo['Survived'] = dfTwo['Survived'].map({1: 'Y', 0: 'N'})

dfTwo.to_csv("testComplete.csv",index = False)

pData = converters.load_any_file("trainComplete.csv")
saver = Saver(classname="weka.core.converters.ArffSaver")
saver.save_file(pData, "processedTrainData.arff")
data = converters.load_any_file("processedTrainData.arff")

multi = MultiFilter()

std = Filter(classname="weka.filters.unsupervised.attribute.Standardize")
#add = Filter(classname='weka.filters.unsupervised.attribute.Add')
multi.filters = [std]
multi.inputformat(data)
data = multi.filter(data)
#print(data)
data.class_is_first()  
cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.2"])
cls.build_classifier(data)
print(cls)


testpData = converters.load_any_file("testComplete.csv")
saver = Saver(classname="weka.core.converters.ArffSaver")
saver.save_file(testpData, "processedTestData.arff")
testData = converters.load_any_file("processedTestData.arff")
#print(testData)


#filteredB = multi.filter(testData)
#print(filteredB)
filteredB = testData
filteredB.class_is_first()
for index, inst in enumerate(filteredB):
    pred = cls.classify_instance(inst)
    dist = cls.distribution_for_instance(inst)
    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
evaluation = Evaluation(data)
#evaluation.crossvalidate_model(cls, filteredB, 10, Random(42))  # 10-fold CV
evaluation.test_model(cls, filteredB)
print(evaluation.summary())

outputPred = pout.buffer_content()
predictions = []
p = []
for line in outputPred.split('\n') :
    m = line.split()
    if((m is not None) & (len(m) > 1)):
        predSplit = m[2].split(':')
        if(len(predSplit) > 1):
            pred = predSplit[1]
            predictions.append(pred)
            print pred                

df1 = pd.DataFrame({'PassengerId':[], 'Survived': []})

df1['Survived'] = predictions
df1['Survived'] = df1['Survived'].map({'Y': 1, 'N': 0})
df1['PassengerId'] = temp['PassengerId']

df1.to_csv("results3.csv",index = False)   

jvm.stop()
