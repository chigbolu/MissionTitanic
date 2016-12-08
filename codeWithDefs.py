from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from time import time
import matplotlib.pyplot as plt
import csv as csv
import numpy as np
import pandas as pd
import math as math

titlesAverages = dict.fromkeys(['Miss','Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])
rareTitles = dict.fromkeys(['Dona', 'Lady', 'the Countess'])
rareTitles = dict.fromkeys(['Dona', 'Lady', 'the Countess','Don', 'Sir'])
rareMan = dict.fromkeys(['Don', 'Sir'])
rareWom = dict.fromkeys(['Lady','the Countess'])
surnames = dict()


def getTitle(name):
    for title in titlesAverages:
        if title + '.' in name:
            return title
    return 'None'

def getSurname(name):
    nameSplit = name.split(',')
    return nameSplit[0]


#only train
def mapTrainStuff(df):
    for title in titlesAverages: 
        rows = df[df.Name.str.contains(' '+title+'.')]
        average = rows['Age'].mean()
        titlesAverages[title] = float(average)

    for index, row in df.iterrows():
        #add title column
        surname = getSurname(row['Name'])
        survived = row['Survived']
        surnames[surname] = survived

    return df   
    
#both
def processTrainAndTestStuff(df):
    df.insert(11,'Title','Mr')

    df.insert(12, 'Family_size', 0)
    for index, row in df.iterrows():
        embark = row['Embarked']
        if(pd.isnull(embark)):
            df.set_value(index,'Embarked','C')      
        age = row['Age']
        if(math.isnan(age)):
            title = getTitle(row['Name'])
            if title in titlesAverages:
                average = titlesAverages[title]
                df.set_value(index, 'Age', average)

        #if age<16:
        #    df.set_value(index, 'Sex', 'child')
        fare = row['Fare'] 
        if(math.isnan(fare)):
            df.set_value(index, 'Fare', 25)


        parCh = int(row ['Parch'])
        sibSp = int(row['SibSp'])
        famSize = parCh + sibSp
        if(famSize == 0):
           famSize = 2
        elif(famSize <= 3):
            famSize = 0
        else:
            famSize = 1
        df.set_value(index, 'Family_size',famSize)
        title = getTitle(row['Name'])
        df.set_value(index, 'Title', title)

    return df   

#only test
def decisionTree(df):
    finalResult = []
    counter = 0;

    for index, row in df.iterrows():
        rowResult = []
        sex = row['Sex']
        sibSp = int(row['SibSp'])
        age = float(row['Age'])
        pClass = int(row['Pclass'])
        emb = row['Embarked']
        parCh = int(row ['Parch'])
        passId = row['PassengerId']
        fare = int(row['Fare'])
        tit  =row['Title']
        name = getSurname(row['Name'])
        familySize = int(row['Family_size'])

        if(sex == 'male'):
            if(age <= 13):
                if(sibSp <= 2):
                    rowResult.append(passId)
                    rowResult.append(1)
                if(sibSp >2):
                    rowResult.append(passId)
                    rowResult.append(0)
            if(age>13):
                rowResult.append(passId)
                rowResult.append(0)     
        if(sex == 'female'):
            if(pClass <= 2):
                rowResult.append(passId)
                rowResult.append(1)
            if(pClass > 2):
                if(emb == 'S'):
                    if(sibSp <= 1):
                        if(sibSp<=0):
                            rowResult.append(passId)
                            rowResult.append(0)
                        if(sibSp>0):
                            if(age <= 38):
                                if(age<=33):
                                    if (parCh<=0):
                                        rowResult.append(passId)
                                        rowResult.append(0)
                                    if(parCh>0):
                                        if(age<=12):
                                            rowResult.append(passId)
                                            rowResult.append(1)
                                        if(age>12):
                                            rowResult.append(passId)
                                            rowResult.append(0)
                                if(age>33):
                                    rowResult.append(passId)
                                    rowResult.append(1)
                            if(age>38):
                                rowResult.append(passId)
                                rowResult.append(0)            
                    if(sibSp>1):             
                        rowResult.append(passId)
                        rowResult.append(0)
                if(emb == 'C'):
                    rowResult.append(passId)
                    rowResult.append(1)
                if(emb == 'Q'):
                    if(parCh<=0):
                        rowResult.append(passId)
                        rowResult.append(1) 
                    else:
                        rowResult.append(passId)
                        rowResult.append(0)
                        
        finalResult.append(rowResult)
    return finalResult


def runModel():
    dfTrain = pd.read_csv('train.csv', header=0)
    dfTest = pd.read_csv('test.csv', header=0)
    
    dfTrain = mapTrainStuff(dfTrain)
    dfTrain = processTrainAndTestStuff(dfTrain)
    dfTest = processTrainAndTestStuff(dfTest)
    print(dfTest)
    dfTrain['Survived'] = dfTrain['Survived'].astype(float)
    print(dfTrain)
    finalResult = decisionTree(dfTest)
    dfFinal = pd.DataFrame(finalResult, columns = list('ps'))
    dfFinal.rename(columns={'p':'PassengerId'}, inplace=True)
    dfFinal.rename(columns={'s':'Survived'}, inplace=True)
    dfFinal.to_csv("results.csv",index = False)

runModel()     
