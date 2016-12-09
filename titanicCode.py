#Python version required: 2.7
#https://www.kaggle.com/c/titanic
#Team: The_Entangled
import csv as csv
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import math as math

titlesAverages = dict.fromkeys(['Miss','Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])

def getTitle(name):
    for title in titlesAverages:
        if title + '.' in name:
            return title
    return 'None'

def assignMother(df):
    df.insert(13,'Mother','1')
    for index,row in df.iterrows():
        if((row['Sex'] == 0) & (row['Parch'] > 0) & (int(row['Age']) > 18) & (getTitle(row['Name']) != 'Miss')):
            df.set_value(index, 'Mother', 1)
        else:
            df.set_value(index, 'Mother', 0)
    return df

def assignChild(df):
    df.insert(14,'Child','1')
    for index,row in df.iterrows():
        if(row['Age'] < 18):
            df.set_value(index, 'Child', 1)
        else:
            df.set_value(index, 'Child', 0)
    return df

#only train
def calculateAgeAvg(df):
    for title in titlesAverages:
        rows = df[df.Name.str.contains(' '+title+'.')]
        average = rows['Age'].mean()
        titlesAverages[title] = float(average)


def addTitleAndFamilySizeAndReplaceEmbarked(df):
    df.insert(11,'Title','Mr')
    df.Fare.fillna(8,inplace=True)
    #df.insert(12, 'Family_size', 0)


    for index,row in df.iterrows():
        age = row['Age']
        if(math.isnan(age)):
            title = getTitle(row['Name'])
            if title == 'None':
                continue
            else:
                average = titlesAverages[title]
                df.set_value(index, 'Age', average)

        #add title column
        #title = getTitle(row['Name'])
    	#df.set_value(index, 'Title', title)
        #add family size
        #parCh = int(row ['Parch'])
    	#sibSp = int(row['SibSp'])
    	#famSize = parCh + sibSp
    	#if(famSize == 0):
    	#	famSize = 2;
    	#elif(famSize <= 3):
    	#	famSize = 0;
    	#else:
    	#	famSize = 1;
    	#df.set_value(index, 'Family_size',famSize)

#replace embarked column
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
    df.Embarked.fillna(1, inplace=True)
    return df

def mapSexToNumerical(df):
    df.loc[df.Sex == 'male', 'Sex'] = 0
    df.loc[df.Sex == 'female', 'Sex'] = 1
    return df;

#implementing results(following) of J48 Decision tree classifier run in weka


def decisionTree(df):
    finalResult = []

    for index, row in df.iterrows():
    	rowResult = []
    	sex = row['Sex']
    	sibSp = int(row['SibSp'])
    	age = float(row['Age'])
    	pClass = int(row['Pclass'])
    	emb = row['Embarked']
    	parCh = int(row ['Parch'])
    	passId = row['PassengerId']
    	#familySize = int(row['Family_size'])
    	if(sex == 0):
    		if(age <= 13):
    			if (sibSp <= 2):
    				rowResult.append(passId)
    				rowResult.append(1)


    			if (sibSp > 2):
    				rowResult.append(passId)
    				rowResult.append(0)

    		if(age > 13):
     			rowResult.append(passId)
    			rowResult.append(0)
    	if(sex > 0):
    		if(pClass <= 2):
    			rowResult.append(passId)
    			rowResult.append(1)

    		if(pClass > 2):
    			if(emb <= 0):
    				rowResult.append(passId)
    				rowResult.append(0)
    			if(emb > 0):
    				if(parCh <= 0):
    					rowResult.append(passId)
    					rowResult.append(1)

    				if(parCh > 0):
    					if(emb <= 1):
    						rowResult.append(passId)
    						rowResult.append(1)
    					if(emb > 1):
    						rowResult.append(passId)
    						rowResult.append(0)
        finalResult.append(rowResult)
    return finalResult

def runModel():
    dfTest = pd.read_csv('test.csv', header=0)
    dfTrain = pd.read_csv('train.csv', header=0)
    calculateAgeAvg(dfTrain)
    dfTrain = addTitleAndFamilySizeAndReplaceEmbarked(dfTrain)
    dfTrain = mapSexToNumerical(dfTrain)
    dfTrain = assignMother(dfTrain)
    dfTrain = assignChild(dfTrain)
    #Only use for train data
    dfTrain['Survived'] = dfTrain['Survived'].map({1: 'Y', 0: 'N'})
    del dfTrain['Cabin']
    del dfTrain['Name']
    del dfTrain['PassengerId']
    del dfTrain['Title']
    del dfTrain['Ticket']
    del dfTrain['Fare']
    dfTrain.to_csv("train2.csv",index = False)

    finalResult = decisionTree(dfTest)

    dfFinal = pd.DataFrame(finalResult, columns = list('ps'))
    dfFinal.rename(columns={'p':'PassengerId'}, inplace=True)
    dfFinal.rename(columns={'s':'Survived'}, inplace=True)
    dfFinal.to_csv("results.csv",index = False)

runModel()
