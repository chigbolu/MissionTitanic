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

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
import weka.core.converters as converters
from weka.core.converters import Loader, Saver
from weka.core.classes import Random


titlesAverages = dict.fromkeys(['Miss','Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])
df = pd.read_csv('train.csv', header=0)
#Calculate averages for each title
#TODO: Calculate average for no titles(None returned by getTitle)
for title in titlesAverages: 
    rows = df[df.Name.str.contains(' '+title+'.')]
    average = rows['Age'].mean()
    titlesAverages[title] = int(average)



def getTitle(name):
    for title in titlesAverages:
        if title + '.' in name:
            return title
    return 'None'

#Replace empty age column with averages
for index,row in df.iterrows():
    age = row['Age']
    if(math.isnan(age)):
        title = getTitle(row['Name'])
        #TODO: Replace age with average for None
        if title == 'None':
            continue
        else:
            average = titlesAverages[title]
            df.set_value(index, 'Age', average)

#replace null embarkment values 

for index, row in df.iterrows():
    embark = row['Embarked']
    if(pd.isnull(embark)):
        df.set_value(index,'Embarked','C')


del df['Name']
del df['PassengerId']
del df['Fare']
del df['Ticket']
del df['Cabin']

df['Survived'] = df['Survived'].map({1: 'N', 0: 'Y'})
df.to_csv("trainCompleteAges.csv",index = False)

jvm.start()
#data = loader.load_file("train2.csv")
#testData = loader.load_file("test2.csv")

pData = converters.load_any_file("trainCompleteAges.csv")

saver = Saver(classname="weka.core.converters.ArffSaver")
saver.save_file(pData, "processedData.arff")

data = converters.load_any_file("processedData.arff")

data.class_is_first()   # set class attribute
cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.2"])
cls.build_classifier(data)
print(cls)

evaluation = Evaluation(data)                     # initialize with prior
evaluation.crossvalidate_model(cls, data, 10, Random(1))  # 10-fold CV

print(evaluation.summary())
print("pctCorrect: " + str(evaluation.percent_correct))
print("incorrect: " + str(evaluation.incorrect))
jvm.stop()

