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



titlesAverages = dict.fromkeys(['Miss','Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])

rareTitles = dict.fromkeys(['Dona', 'Lady', 'the Countess'])
rareTitles = dict.fromkeys(['Dona', 'Lady', 'the Countess','Don', 'Sir'])
rareMan = dict.fromkeys(['Don', 'Sir'])
rareWom = dict.fromkeys(['Lady','the Countess'])



df = pd.read_csv('train.csv', header=0)
#dfTest = pd.read_csv('test.csv', header=0)
#Calculate averages for each title
#TODO: Calculate average for no titles(None returned by getTitle)
for title in titlesAverages: 
    rows = df[df.Name.str.contains(' '+title+'.')]
    average = rows['Age'].mean()
    titlesAverages[title] = float(average)



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
        else:
            average = titlesAverages[title]
            df.set_value(index, 'Age', average)

    #if age<16:
     # df.set_value(index, 'Sex', 'child')

#replace null embarkment values 

for index, row in df.iterrows():
	embark = row['Embarked']
	if(pd.isnull(embark)):
		df.set_value(index,'Embarked','C')


df.insert(11,'Title','Mr')



df.Fare.fillna(8,inplace=True)
df.insert(12, 'Family_size', 0)

# Cabin
df.Cabin.fillna('0', inplace=True)
df.loc[df.Cabin.str[0] == 'A', 'Cabin'] = 1
df.loc[df.Cabin.str[0] == 'B', 'Cabin'] = 2
df.loc[df.Cabin.str[0] == 'C', 'Cabin'] = 3
df.loc[df.Cabin.str[0] == 'D', 'Cabin'] = 4
df.loc[df.Cabin.str[0] == 'E', 'Cabin'] = 5
df.loc[df.Cabin.str[0] == 'F', 'Cabin'] = 6
df.loc[df.Cabin.str[0] == 'G', 'Cabin'] = 7
df.loc[df.Cabin.str[0] == 'T', 'Cabin'] = 8
    # Embarked
df.loc[df.Embarked == 'S', 'Embarked'] = 0
df.loc[df.Embarked == 'C', 'Embarked'] = 1
df.loc[df.Embarked == 'Q', 'Embarked'] = 2
df.Embarked.fillna(1, inplace=True)
		
del df['Cabin']


#add title and family size columns
for index, row in df.iterrows():		
	title = getTitle(row['Name'])
	df.set_value(index, 'Title', title)	
	parCh = int(row ['Parch'])
	sibSp = int(row['SibSp'])
	famSize = parCh + sibSp
	if(famSize == 0):
		famSize = 2;	
	elif(famSize <= 3):
		famSize = 0;
	else:
		famSize = 1;
	df.set_value(index, 'Family_size',famSize)


# file to run in weka ONLY FOR TRAIN DATA
# TO REMOVE WHEN RUNNING TEST DATA

#df['Survived'] = df['Survived'].map({1: 'Y', 0: 'N'})


#only when running train data  ---------------------

df.loc[df.Survived == 0, 'Survived'] = 'N'
df.loc[df.Survived == 1,'Survived' ] = 'Y'

df.loc[df.Sex == 'male', 'Sex'] = 0
df.loc[df.Sex == 'female', 'Sex'] = 1

#---------------------------------------

df.to_csv("testWeka.csv",index = False)





#implementing results(following) of J48 Decision tree classifier run in weka



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
	familySize = int(row['Family_size'])
	if(sex == 0):
		if(age <= 9):
			if(sibSp <= 2):
				if(parCh == 0):
					rowResult.append(passId)
					rowResult.append(0)
				if(parCh > 0): 
					rowResult.append(passId)
					rowResult.append(1)
			if(sibSp > 2):
				rowResult.append(passId)
				rowResult.append(0)

		if(age > 9):     #this condition could be improved
 			rowResult.append(passId)
			rowResult.append(0)
	if(sex > 0):
		if(pClass <= 2):
			rowResult.append(passId)
			rowResult.append(1)

		if(pClass > 2):	
			if(familySize <= 3):
				rowResult.append(passId)
				rowResult.append(1)
			if(familySize > 3):
				rowResult.append(passId)
				rowResult.append(0)

	finalResult.append(rowResult)



#for row in finalResult:
#	print(row)

#count = 0
#ind = 0

#loop to check the results in the training test
#for index,row in df.iterrows():
#	surv = row['Survived']
#	print "surv value" , surv
#	print "final result value", finalResult[ind][1]
#	if(finalResult[ind][1] == surv):
#		count += 1
#	ind += 1

#print "The accuracy of J48 is:", float(count)/float(len(finalResult))



finalFile = np.asarray(finalResult)	
np.savetxt("testResults.csv",finalFile,delimiter = ",")

#print(count)
#dfFinal.to_csv("testResults.csv",index = False)

				


#here some graph implementation code if needed


		#import matplotlib.pyplot as plt

		#df['Age'] = pd.to_numeric(df['Age'])
		#df['Fare'] = pd.to_numeric(df['Fare'])

		#ax = df[['Age','Fare']].plot(kind='hist', title ="Age Survived graph", figsize=(15, 10), legend=True, fontsize=12)
		#ax.set_xlabel("Age", fontsize=12)
		#ax.set_ylabel("Fare", fontsize=12)
		#plt.show()



		
		

























