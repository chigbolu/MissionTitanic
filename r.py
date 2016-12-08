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
dfTest = pd.read_csv('test.csv', header=0)
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


def getSurname(name):
    nameSplit = name.split(',')
    return nameSplit[0]
    

df.insert(11,'Title','Mr')
surnames = dict()
for index, row in df.iterrows():
    #add title column
	title = getTitle(row['Name'])
	df.set_value(index, 'Title', title)

	surname = getSurname(row['Name'])
	survived = row['Survived']
	surnames[surname] = survived

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

    if age<16:
      df.set_value(index, 'Sex', 'child')

#replace null embarkment values 

for index, row in df.iterrows():
	embark = row['Embarked']
	if(pd.isnull(embark)):
		df.set_value(index,'Embarked','C')



df.insert(12, 'Family_size', 0)
		
#add title column
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


df['Survived'] = df['Survived'].map({1: 'Y', 0: 'N'})

# file to run in weka ONLY FOR TRAIN DATA
# TO REMOVE WHEN RUNNING TEST DATA

df.to_csv("trainWeka.csv",index = False)





#implementing results(following) of J48 Decision tree classifier run in weka

#Sex = male
#|   Pclass <= 1
#|   |   SibSp <= 0
#|   |   |   Age <= 52
#|   |   |   |   Fare <= 26: N (9.0)
#|   |   |   |   Fare > 26
#|   |   |   |   |   Fare <= 27: Y (13.0/2.0)
#   |   |   |   |   Fare > 27: N (46.0/15.0)
#|   |   |   Age > 52: N (19.0/2.0)
#|   |   SibSp > 0
#|   |   |   Embarked <= 0: N (21.0/8.0)
#|   |   |   Embarked > 0
#|   |   |   |   Fare <= 93.5: Y (8.0/2.0)
#|   |   |   |   Fare > 93.5: N (3.0)
#|   Pclass > 1: N (418.0/46.0)
#Sex = female
#|   Pclass <= 2: Y (157.0/8.0)
#|   Pclass > 2
#|   |   Fare <= 24.15
#|   |   |   Embarked <= 1
#|   |   |   |   Title = Mr: N (0.0)
#|   |   |   |   Title = Mrs
#|   |   |   |   |   SibSp <= 0: Y (16.0/5.0)
#|   |   |   |   |   SibSp > 0
#|   |   |   |   |   |   Fare <= 15.1: N (5.0)
#|   |   |   |   |   |   Fare > 15.1
#|   |   |   |   |   |   |   Age <= 32: N (6.0/2.0)
#|   |   |   |   |   |   |   Age > 32: Y (5.0)
#|   |   |   |   Title = Miss
#|   |   |   |   |   Fare <= 7.925
#|   |   |   |   |   |   Age <= 27: Y (16.0/5.0)
#|   |   |   |   |   |   Age > 27: N (3.0)
#|   |   |   |   |   Fare > 7.925: N (17.0/3.0)
#|   |   |   |   Title = Master: N (0.0)
#|   |   |   |   Title = Don: N (0.0)
#|   |   |   |   Title = Rev: N (0.0)
#|   |   |   |   Title = Dr: N (0.0)
#|   |   |   |   Title = None: N (0.0)
#|   |   |   |   Title = Ms: N (0.0)
#|   |   |   |   Title = Major: N (0.0)
#|   |   |   |   Title = Lady: N (0.0)
#|   |   |   |   Title = Sir: N (0.0)
#|   |   |   |   Title = Col: N (0.0)
#|   |   |   |   Title = Capt: N (0.0)
#|   |   |   |   Title = the Countess: N (0.0)
#|   |   |   |   Title = Jonkheer: N (0.0)
#|   |   |   Embarked > 1
#|   |   |   |   Parch <= 0
#|   |   |   |   |   Fare <= 7.65: N (2.0)
#|   |   |   |   |   Fare > 7.65: Y (27.0/4.0)
#|   |   |   |   Parch > 0: N (2.0)
#|   |   Fare > 24.15: N (15.0/1.0)
#Sex = child
#|   SibSp <= 2: Y (56.0/9.0)
#|   SibSp > 2: N (27.0/2.0)


finalResult = []

for index, row in dfTest.iterrows():
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

	if((name in surnames) & (surnames[name]==1)):
			rowResult.append(passId)
			rowResult.append(1)

	elif(sex == 'male'):
		if(pClass<=1):
			if(sibSp == 0):
				if(age <= 52):
					if(fare <= 26):
						rowResult.append(passId)
						rowResult.append(0)

					if(fare > 26):
						if(fare <= 27):
							rowResult.append(passId)
							rowResult.append(1)
						if(fare > 27):
							rowResult.append(passId)
							rowResult.append(0)

				if(age > 52):
					rowResult.append(passId)
					rowResult.append(0)
			if(sibSp > 0):
				if(emb == 0):
					rowResult.append(passId)
					rowResult.append(0)
				if(emb > 0):
					if(fare <= 93.5):
						rowResult.append(passId)
						rowResult.append(0)
					if(fare > 93.5):
						rowResult.append(passId)
						rowResult.append(1)
		if(pClass > 1):
			rowResult.append(passId)
			rowResult.append(0)
	elif(sex == 'female'):
		if(pClass <= 2):
			rowResult.append(passId)
			rowResult.append(1)
		if(pClass > 2):
			if(fare <= 24.15):
				if(emb <= 1):
					if(tit == 'Mrs'):
						if(sibSp == 0):
							rowResult.append(passId)
							rowResult.append(1)
						if(sibSp > 0):
							if(fare <= 15.1):
								rowResult.append(passId)
								rowResult.append(0)
							if(fare > 15.1):
								if(age <= 32):
									rowResult.append(passId)
									rowResult.append(0)
								if(age > 32):
									rowResult.append(passId)
									rowResult.append(1)
					if(tit == 'Miss'):
						if(fare < 7.925):
							if(age <= 27):
								rowResult.append(passId)
								rowResult.append(1)
							if(age > 27):
								rowResult.append(passId)
								rowResult.append(0)
						if(fare > 7.925):
							rowResult.append(passId)
							rowResult.append(0)
				
				if(emb > 1):
					if(parCh == 0):
						if(fare <= 7.65):
							rowResult.append(passId)
							rowResult.append(0)
						if(fare > 7.65):
							rowResult.append(passId)
							rowResult.append(1)
					if(parCh > 0):
						rowResult.append(passId)
						rowResult.append(0)
			if(fare > 24.15):
				rowResult.append(passId)
				rowResult.append(0)
	elif(sex == 'child'):
		if(sibSp <= 2):
			rowResult.append(passId)
			rowResult.append(1)
		if(sibSp > 2):
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




dfFinal = pd.DataFrame(finalResult, columns = list('ps'))
#print(dfFinal)

#df.rename(0, 'PassengerId')
#df.rename(1, 'Survived')


#finalFile = np.asarray(finalResult)	
#np.savetxt("testResults.csv",finalFile,delimiter = ",")

dfFinal.to_csv("testResults.csv",index = False)

				


#here some graph implementation code if needed


		#import matplotlib.pyplot as plt

		#df['Age'] = pd.to_numeric(df['Age'])
		#df['Fare'] = pd.to_numeric(df['Fare'])

		#ax = df[['Age','Fare']].plot(kind='hist', title ="Age Survived graph", figsize=(15, 10), legend=True, fontsize=12)
		#ax.set_xlabel("Age", fontsize=12)
		#ax.set_ylabel("Fare", fontsize=12)
		#plt.show()



		
		























