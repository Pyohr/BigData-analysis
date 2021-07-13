import pandas as pd
import numpy as np
from collections import Counter

data = pd.read_csv("/home/pyo/Downloads/module4/train_20000.csv", names=['ID','product','order','reorder'])

def loadDataSet(data):
	data=data.drop(['order','reorder'],axis =1)
	cart = []
	ID = data['ID'][0]
	temp = []
	for i in range(0,20000):
		if(ID==data['ID'][i]):
			 temp.append(data['product'][i])
		else:
			 cart.append(temp)
			 temp=[]
			 ID=data['ID'][i]
			 temp.append(data['product'][i])
	return cart

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList    


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] 
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] 
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)




data['count']=1

data_sort = data['count'].groupby(data['product']).sum()
data_sort = data_sort.sort_values(ascending=False)

cartdata = loadDataSet(data)
L,suppData= apriori(cartdata, minSupport=0.002)
rules= generateRules(L,suppData, minConf=0.02)


f= open('/home/pyo/Downloads/module4/test.csv')
data=f.readlines()
data_t=[]
test_data=[]
try:
	for line in data:
		data_t=line[:-2].split(',')
		data_t = list(map(int, data_t))
		test_data.append(frozenset(data_t))
		
finally:
	f.close()


results = []
for i in range(0,len(test_data)):
	temp = []
	for j in range(0,len(rules)):
		if(test_data[i].issuperset(rules[j][0])):
			temp.append([rules[j][1],rules[j][2]])
	result=sorted(temp, key =lambda conf:conf[1],reverse=True)
	results.append(result)	


data_sort_array = data_sort.index.values
most_5 = [data_sort_array[0],data_sort_array[1],data_sort_array[2],data_sort_array[3],data_sort_array[4]]


for i in range(0,len(results)):
	if(len(results[i])>5):
		results[i]=[list(results[i][0][0]),list(results[i][1][0]),list(results[i][2][0]),list(results[i][3][0]),list(results[i][4][0])]
	elif(len(results[i])==4):
		results[i]=[list(results[i][0][0]),list(results[i][1][0]),list(results[i][2][0]),list(results[i][3][0]),most_5[0]]
	elif(len(results[i])==3):
		results[i]=[list(results[i][0][0]),list(results[i][1][0]),list(results[i][2][0]),most_5[0],most_5[1]]
	elif(len(results[i])==2):
		results[i]=[list(results[i][0][0]),list(results[i][1][0]),most_5[0],most_5[1],most_5[2]]
	elif(len(results[i])==1):
		results[i]=[list(results[i][0][0]),most_5[0],most_5[1],most_5[2],most_5[3]]
	else:
		results[i]=most_5



submission = pd.DataFrame(results)
submission.to_csv('m4_result.csv', header=False,index=False)
