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
    calculateAgeAvg(dfTest)
    dfTest = addTitleAndFamilySizeAndReplaceEmbarked(dfTest)
    dfTest = mapSexToNumerical(dfTest)
    del dfTest['Cabin']
    finalResult = decisionTree(dfTest)

    dfFinal = pd.DataFrame(finalResult, columns = list('ps'))
    dfFinal.rename(columns={'p':'PassengerId'}, inplace=True)
    dfFinal.rename(columns={'s':'Survived'}, inplace=True)
    dfFinal.to_csv("results.csv",index = False)

runModel()
