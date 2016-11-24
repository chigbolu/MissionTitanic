
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
import numpy as np
import matplotlib.pyplot as plt

trainData = csv.reader(open('./train.csv','rb'))
#Open test data and call next function to skip the first line(header)
trainData = csv.reader(open('./train2.csv','rb'))
header = trainData.next();

#Loop through csv rows and add to data 
data = []
for row in trainData:
    data.append(row)
#Convert to numpy array(more efficient than python lists)
data = np.array(data);

travellingFirstFromS = []
travellingFirstFromC = []
travellingFirstFromQ = []
travellingSecondFromS = []
travellingSecondFromC = []
travellingSecondFromQ = []
travellingThirdFromS = []
travellingThirdFromC = []
travellingThirdFromQ = []

ticketCostsFirstS = []
ticketCostsSecondS = []
ticketCostsThirdS = []
ticketCostsFirstC = []
ticketCostsSecondC = []
ticketCostsThirdC = []
ticketCostsFirstQ = []
ticketCostsSecondQ = []
ticketCostsThirdQ = []

#obtain row of each class(1,2,3) and store them in different arrays
for row in data:

	if (row[11] == "S") & (row[2] == "1") :
		travellingFirstFromS.append(row)
	if (row[11] == "C") & (row[2] == "1") :
		travellingFirstFromC.append(row)
	if (row[11] == "Q") & (row[2] == "1" ):
		travellingFirstFromQ.append(row)
	if (row[11] == "S") & (row[2] == "2") :
		travellingSecondFromS.append(row)
	if (row[11] == "C") & (row[2] == "2") :
		travellingSecondFromC.append(row)
	if (row[11] == "Q") & (row[2] == "2" ):
		travellingSecondFromQ.append(row)
	if (row[11] == "S") & (row[2] == "3") :
		travellingThirdFromS.append(row)
	if (row[11] == "C") & (row[2] == "3") :
		travellingThirdFromC.append(row)
	if (row[11] == "Q") & (row[2] == "3" ):
		travellingThirdFromQ.append(row)

# obtain the count of each ticket, store the ticket instance as key

dictFirstS = {}
for row in travellingFirstFromS:
	instance = row[8]
	if(instance in dictFirstS) :
		dictFirstS[instance] += 1;
	else :
		dictFirstS[instance] = 1;


dictSecondS = {}
for row in travellingSecondFromS:
	instance = row[8]
	if(instance in dictSecondS) :
		dictSecondS[instance] += 1;
	else :
		dictSecondS[instance] = 1;

dictThirdS = {}
for row in travellingThirdFromS:
	instance = row[8]
	if(instance in dictThirdS) :
		dictThirdS[instance] += 1;
	else :
		dictThirdS[instance] = 1;



dictFirstC = {}
for row in travellingFirstFromC:
	instance = row[8]
	if(instance in dictFirstC) :
		dictFirstC[instance] += 1;
	else :
		dictFirstC[instance] = 1;


dictSecondC = {}
for row in travellingSecondFromC:
	instance = row[8]
	if(instance in dictSecondC) :
		dictSecondC[instance] += 1;
	else :
		dictSecondC[instance] = 1;

dictThirdC = {}
for row in travellingThirdFromC:
	instance = row[8]
	if(instance in dictThirdC) :
		dictThirdC[instance] += 1;
	else :
		dictThirdC[instance] = 1;



dictFirstQ = {}
for row in travellingFirstFromQ:
	instance = row[8]
	if(instance in dictFirstQ) :
		dictFirstQ[instance] += 1;
	else :
		dictFirstQ[instance] = 1;


dictSecondQ = {}
for row in travellingSecondFromQ:
	instance = row[8]
	if(instance in dictSecondQ) :
		dictSecondQ[instance] += 1;
	else :
		dictSecondQ[instance] = 1;

dictThirdQ = {}
for row in travellingThirdFromQ:
	instance = row[8]
	if(instance in dictThirdQ) :
		dictThirdQ[instance] += 1;
	else :
		dictThirdQ[instance] = 1;




#get the price of a single ticket embark S
for row in travellingFirstFromS:
	ticket = row[8]
	
	if(float(row[9]) != 0. ):
		price = float(row[9])
        if (ticket in dictFirstS):
                ticketCostsFirstS.append(price/dictFirstS.get(ticket))


#get the price of a single ticket embark S
for row in travellingSecondFromS:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictSecondS):
		         ticketCostsSecondS.append(price/dictSecondS.get(ticket))

#get the price of a single ticket embark S
for row in travellingThirdFromS:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
        if (ticket in dictThirdS):
                ticketCostsThirdS.append(price/dictThirdS.get(ticket))



#get the price of a single ticket embark C
for row in travellingFirstFromC:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictFirstC):
                	ticketCostsFirstC.append(price/dictFirstC.get(ticket))

#get the price of a single ticket embark C
for row in travellingSecondFromC:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictSecondC):
                	ticketCostsSecondC.append(price/dictSecondC.get(ticket))


#get the price of a single ticket embark C
for row in travellingThirdFromC:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictThirdC):
                	ticketCostsThirdC.append(price/dictThirdC.get(ticket))



#get the price of a single ticket embark Q
for row in travellingFirstFromQ:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictFirstQ):
                	ticketCostsFirstQ.append(price/dictFirstQ.get(ticket))

#get the price of a single ticket embark Q
for row in travellingSecondFromQ:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictSecondQ):
                	ticketCostsSecondQ.append(price/dictSecondQ.get(ticket))

#get the price of a single ticket embark Q
for row in travellingThirdFromQ:
	ticket = row[8]
	if(float(row[9]) != 0.):
		price = float(row[9])
		if (ticket in dictThirdQ):
                	ticketCostsThirdQ.append(price/dictThirdQ.get(ticket))

#print("First S")
#print(sorted(ticketCostsFirstS))
#print("Second S")
#print(ticketCostsSecondS)
#print("Third S")
#print(ticketCostsThirdS)
#print("First C")
#print(ticketCostsFirstC )
#print("Second C")
#print(ticketCostsSecondC)
#print("Third C")
#print(ticketCostsThirdC)
#print("First Q")
#print(ticketCostsFirstQ )
#print("Second Q")
#print(ticketCostsSecondQ )
#print("Third C")
#print(ticketCostsThirdQ )

ticketAverageCosts = []

ticketAverageCosts.append(sum(ticketCostsFirstS)/len(ticketCostsFirstS))

ticketAverageCosts.append(sum(ticketCostsSecondS)/len(ticketCostsSecondS))

ticketAverageCosts.append(sum(ticketCostsThirdS)/len(ticketCostsThirdS))


ticketAverageCosts.append(sum(ticketCostsFirstC)/len(ticketCostsFirstC))

ticketAverageCosts.append(sum(ticketCostsSecondC)/len(ticketCostsSecondC))

ticketAverageCosts.append(sum(ticketCostsThirdC)/len(ticketCostsThirdC))

ticketAverageCosts.append(sum(ticketCostsFirstQ)/len(ticketCostsFirstQ))

ticketAverageCosts.append(sum(ticketCostsSecondQ)/len(ticketCostsSecondQ))

ticketAverageCosts.append(sum(ticketCostsThirdQ)/len(ticketCostsThirdQ))

#print(ticketAverageCosts)

#stefan
#calculate family sizes

familySizes = []
dictFamilies = {}
for row in travellingFirstFromS:
	instance = row[3]
	if(instance in dictFamilies) :
		var = "hello"
	else :
		dictFamilies[instance] = int(row[6])+int(row[7]);
	
# family names and sizes printed nicely with iterations

for key,value in dictFamilies.iteritems():
	print key,value


# obtain children price ticket

#get the price of a single ticket embark S

singleTicket1S = []
for row in travellingFirstFromS:
	if(row[8] in dictFirstS):
		if(dictFirstS[row[8]] == 1):
			fare = float(row[9])
			singleTicket1S.append(fare)

print("Single 1s")
print(sum(singleTicket1S)/len(singleTicket1S))

singleTicket1C = []
for row in travellingFirstFromC:
	if(row[8] in dictFirstC):
		if(dictFirstC[row[8]] == 1):
			fare = float(row[9])
			singleTicket1C.append(fare)

print("Single 1c")
print(sum(singleTicket1C)/len(singleTicket1C))

singleTicket1Q = []
for row in travellingFirstFromQ:
	if(row[8] in dictFirstQ):
		if(dictFirstQ[row[8]] == 1):
			fare = float(row[9])
			singleTicket1Q.append(fare)

#print("Single 1q")
#print(sum(singleTicket1Q)/len(singleTicket1Q))

singleTicket2S = []

for row in travellingSecondFromS:
	fA = []
	if(row[8] in dictSecondS):
		if(dictSecondS[row[8]] == 1):
			if(row[5] != "" and row[9] != ""):
				fare = float(row[9])
				age = float(row[5])
				fA.append(age)
				fA.append(fare)
				singleTicket2S.append(fA)
			

#print("Single 2s")
#print(sum(singleTicket2S)/len(singleTicket2S))

singleTicket2C = []
for row in travellingSecondFromC:
	if(row[8] in dictSecondC):
		if(dictSecondC[row[8]] == 1):
			fare = float(row[9])
			singleTicket2C.append(fare)
print("Single 2C")
print(sum(singleTicket2C)/len(singleTicket2C))


singleTicket2Q = []
for row in travellingSecondFromQ:
	if(row[8] in dictSecondQ):
		if(dictSecondQ[row[8]] == 1):
			fare = float(row[9])
			singleTicket2Q.append(fare)
print("Single 2q")
print(sum(singleTicket2Q)/len(singleTicket2Q))

singleTicket3S = []
for row in travellingThirdFromS:
	if(row[8] in dictThirdS):
		if(dictThirdS[row[8]] == 1):
			fare = float(row[9])
			singleTicket3S.append(fare)

print("Single 3s")
print(sum(singleTicket3S)/len(singleTicket3S))

singleTicket3C = []
for row in travellingThirdFromC:
	if(row[8] in dictThirdC):
		if(dictThirdC[row[8]] == 1):
			fare = float(row[9])
			singleTicket3C.append(fare)

print("Single 3c")
print(sum(singleTicket3C)/len(singleTicket3C))

singleTicket3Q = []
for row in travellingThirdFromQ:
	if(row[8] in dictThirdQ):
		if(dictThirdQ[row[8]] == 1):
			fare = float(row[9])
			singleTicket3Q.append(fare)

print("Single 3Q")
print(sum(singleTicket3Q)/len(singleTicket3Q))


doubleValues = []

# kmeans does not give great results for the age problem
for row in travellingFirstFromS:
	if(row[5] is not None and row[9] is not None):
		ageFare = []
		try:
			ageFare.append(int(row[5]))
			#ageFare.append(int(row[1]))
			#ageFare.append(int(row[6]))
			#ageFare.append(int(row[7]))
			ageFare.append(float(row[9]))
			doubleValues.append(ageFare)
		except ValueError,e:
			print "Error --> " ,e
			

print("Here the double values")
print(doubleValues)

kmeans = KMeans(n_clusters=10, random_state=0).fit(doubleValues)
kmeans.labels_
print(kmeans.cluster_centers_)


doubleValues = []

# kmeans does not give great results for the age problem
for row in singleTicket2S:
	ageFare = []
	ageFare.append(int(row[0]))
			#ageFare.append(int(row[1]))
			#ageFare.append(int(row[6]))
			#ageFare.append(int(row[7]))
	ageFare.append(float(row[1]))
	doubleValues.append(ageFare)
			

print("Here the double values")
print(doubleValues)

kmeans = KMeans(n_clusters=15, random_state=0).fit(doubleValues)

print(kmeans.cluster_centers_)

x = []
y = []
xy = []

# kmeans does not give great results for the age problem
for row in singleTicket2S:
	ageFare = []
	x.append(int(row[0]))
			#ageFare.append(int(row[1]))
			#ageFare.append(int(row[6]))
			#ageFare.append(int(row[7]))
	y.append(float(row[1]))
		#	doubleValues.append(ageFare)
		
xy.append(y)
print(singleTicket2S)

plt.plot(x,y, 'ro')
plt.axis([0, 90, 0, 50])
plt.show()
#y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

#plt.fill(x, y, 'r')
#plt.grid(True)
#plt.show()





