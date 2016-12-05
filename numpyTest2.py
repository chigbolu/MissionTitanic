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
df = pd.read_csv('test.csv', header=0)
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


df.to_csv("trainCompleteAges.csv",index = False)


#implementing results(following) of J48 Decision tree classifier run in weka

#model has 7 attributes , replace embarkment with family size

#Sex = male
#|   Age <= 9
#|   |   SibSp <= 2
#|   |   |   Parch <= 0: N (8.19/0.99)
#|   |   |   Parch > 0: Y (18.21/0.07)
#|   |   SibSp > 2: N (14.35/1.0)
#|   Age > 9: N (536.24/88.87)
#Sex = female
#|   Pclass <= 2: Y (170.0/9.0)
#|   Pclass > 2
#|   |   Family_size <= 3: Y (117.0/48.0)
#|   |   Family_size > 3: N (27.0/3.0)


#implementation in the code, to predict surv and non surv

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
	if(sex == 'male'):
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
	if(sex == 'female'):
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

count = 0
ind = 0

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
				


#here some graph implementation code if needed


		#import matplotlib.pyplot as plt

		#df['Age'] = pd.to_numeric(df['Age'])
		#df['Fare'] = pd.to_numeric(df['Fare'])

		#ax = df[['Age','Fare']].plot(kind='hist', title ="Age Survived graph", figsize=(15, 10), legend=True, fontsize=12)
		#ax.set_xlabel("Age", fontsize=12)
		#ax.set_ylabel("Fare", fontsize=12)
		#plt.show()



		
		

























