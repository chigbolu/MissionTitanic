
#Python version required: 2.7
#https://www.kaggle.com/c/titanic
import csv as csv
import numpy as np
from time import time
import numpy as np

trainData = csv.reader(open('./train.csv','rb'))
#Open test data and call next function to skip the first line(header)
header = trainData.next();

#Loop through csv rows and add to data 
data = []
for row in trainData:
    data.append(row)
#Convert to numpy array(more efficient than python lists)
data = np.array(data);
titlesAverages = dict.fromkeys(['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Mr', 'Master', 'Ms', 'Mrs'])
for title in titles:
    #Method 1 : where search string(title) can be in any column
    #use in function to check if the title is in the string
    rows = np.where(data[:,4] == title)
    newData = data[rows]
    average = np.mean(newData)
    titlesAverages[title] = average

