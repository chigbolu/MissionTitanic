#Python version required: 2.7
#https://www.kaggle.com/c/titanic
import csv as csv
import numpy as np

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
	if(float(row[9]) != 0.):
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


print(ticketCostsFirstS)
print(ticketCostsSecondS)
print(ticketCostsThirdS)
#print(ticketCostsFirstC )
#print(ticketCostsSecondC)
#print(ticketCostsThirdC)
#print(ticketCostsFirstQ )
#print(ticketCostsSecondQ )
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

print(ticketAverageCosts)

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

singleTicketS = []
for row in travellingFirstFromS:
	if(row[8] in dictFirstS):
		if(dictFirstS[row[8]] == 1):
			fare = row[9]	
			singleTicketS.append(fare)


print(singleTicketS)







#print(travellingFirstFromC)



#to do: Store familes by surname possibly
#estimate how many members of family survive
	#store families by second name and family size





#print(ticketInstances)
#print(travellingFirstFromC)


#travellingFirstFromS = travellingFromS[0::,2] == "1"
#travellingSecondFromS = travellingFromS[0::,2] == "2"
#travellingThirdFromS = travellingFromS[0::,2] == "3"

#travellingFirstFromC = travellingFromC[0::,2] == "1"
#travellingSecondFromC = travellingFromC[0::,2] == "2"
#travellingThirdFromC = travellingFromC[0::,2] == "3"

#travellingFirstFromQ = travellingFromC[0::,2] == "1"
#travellingSecondFromQ = travellingFromC[0::,2] == "2"
#travellingThirdFromQ = travellingFromC[0::,2] == "3"

#how to group them?
#sort tickets
#count the instances
#if one - store in singles
#if two+ stor in families
#count out all people with fare 0
#add float convert
