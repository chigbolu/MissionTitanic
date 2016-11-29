
#Python version required: 2.7
#https://www.kaggle.com/c/titanic
import csv as csv
from time import time
import pandas as pd
import math as math
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
df.to_csv("train.csv")
