
#Python version required: 2.7
#https://www.kaggle.com/c/titanic
import csv as csv
from time import time
import pandas as pd
import math as math
titlesAverages = dict.fromkeys(['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])
df = pd.read_csv('train.csv', header=0)
#Calculate averages for each title
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
        if title == 'None':
            continue
        average = titlesAverages[title]
        row['Age'] = average
df.to_csv("train.csv")
