
#Python version required: 2.7
#https://www.kaggle.com/c/titanic
import csv as csv
from time import time
import pandas as pd

titlesAverages = dict.fromkeys(['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])
df = pd.read_csv('train.csv', header=0)
#Calculate averages for each title
for title in titlesAverages: 
    rows = df[df.Name.str.contains(' '+title+'.')]
    average = rows['Age'].mean()
    titlesAverages[title] = int(average)

