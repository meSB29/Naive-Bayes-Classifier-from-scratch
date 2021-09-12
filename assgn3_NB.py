from copy import deepcopy
import math
from math import factorial
from time import time
import random
import sys

t0=time()
def load_dataset(filename):
	file_string=""
	with open(filename,'r') as file:
		file_string=file_string+file.read()
	file_string=file_string.lower()
	file_string=file_string.replace('\W',' ')
	file_list=list(file_string.split('\n'))
	dataset=[]
	for var in file_list:
		dataset.append(var.split('\t'))
	return dataset

def multivariateNB(data,k):
	accuracy=0
	folds=[]
	for r in range(k):
		folds.append(data[(len(data)*r//k):(min(len(data),len(data)*(r+1)//k))])
	for fold in range(len(folds)):
		# PREPARING TRAIN SET AND TEST SET
		train_data=[]
		for tf in range(len(folds)):
			if tf!=fold:
				train_data.extend(folds[tf])
		test_data=[i for i in folds[fold]]
		
		N=len(train_data)
		Class_list=list(set([str(i[0]).lower() for i in train_data]))
		Class={}
		for i in range(len(Class_list)):
			Class[Class_list[i]]=i
		#CALCULATING PRIOR PROBABILITY
		prior={i:0 for i in Class.keys()}
		for i in train_data:
			prior[str(i[0]).lower()]=prior[str(i[0]).lower()]+1
		
		V={}
		vocab_set=0
		for i in train_data:
			vocab=list(i[1].split(' '))
			vocab=list(set(vocab))
			vocab_set=vocab_set+len(vocab)
			for j in vocab:
				if j not in V.keys():
					V[j.lower()]=[0 for i in range(len(Class))]
				V[j.lower()][Class[i[0]]]=V[j.lower()][Class[i[0]]]+1
		
		for i in V.keys():
			for j in prior.keys():
				V[i][Class[j]]=(V[i][Class[j]]+1)/(prior[j]+vocab_set)
		
		for i in prior.keys():
			prior[i]=prior[i]/N

	# CLASSIFICATION USING NAIVE BAYES MULTIVARIATE
		
		predictions=[]
		for value in range(len(test_data)):
			test=list(test_data[value][1].split(' '))
			test=[i.lower() for i in test]
			score={}
			maxm=-100000
			res_class=''
			for c in Class.keys():
				score[c]=math.log2(prior[c])/math.log2(math.exp(1))
				for t in V.keys():
					if t in test:
						score[c]=score[c]+(math.log2(V[t][Class[c]])/math.log2(math.exp(1)))
					else:
						score[c]=score[c]+(math.log2(1-V[t][Class[c]])/math.log2(math.exp(1)))
				if score[c]>maxm:
					maxm=score[c]
					res_class=c
			
			predictions.append([test_data,test_data[value][0],res_class])
		
		correct_class=[1 for i in predictions if i[1]==i[2]]
		print('Accuracy of fold ',fold+1,': ',round(len(correct_class)*100/len(test_data),3))
		accuracy=accuracy+len(correct_class)/len(test_data)
	return accuracy/k

def multinomialNB(data,k):
	accuracy=0
	folds=[]
	for r in range(k):
		folds.append(data[(len(data)*r//k):(min(len(data),len(data)*(r+1)//k))])
	for fold in range(len(folds)):
		train_data=[]
		for tf in range(len(folds)):
			if tf!=fold:
				train_data.extend(folds[tf])
		test_data=[i for i in folds[fold]]
		
		N=len(train_data)
		Class_list=list(set([str(i[0]).lower() for i in train_data]))
		Class={}
		for i in range(len(Class_list)):
			Class[Class_list[i]]=i
		
		prior={i:0 for i in Class.keys()}
		for i in train_data:
			prior[str(i[0]).lower()]=prior[str(i[0]).lower()]+1
		
		V={}
		mega_document=['' for i in prior.keys()]
		for i in train_data:
			mega_document[Class[i[0]]]+=i[1]

		
		vocab_set=0
		for i in train_data:
			vocab=i[1].split(' ')
			vocab_set=vocab_set+len(list(set(vocab)))
			for j in list(set(vocab)):
				if j not in V.keys():
					V[j.lower()]=[0 for i in range(len(Class))]
				if V[j.lower()][Class[i[0]]]==0:
					V[j.lower()][Class[i[0]]]=mega_document[Class[i[0]]].count(j.lower())

		for i in V.keys():
			for j in prior.keys():
				V[i][Class[j]]=(V[i][Class[j]]+1)/(prior[j]+vocab_set)

		for i in prior.keys():
			prior[i]=prior[i]/N	
	# CLASSIFICATION USING NAIVE BAYES MULTINOMIAL
		predictions=[]
		for value in range(len(test_data)):
			test=list(test_data[value][1].split(' '))
			test=[i.lower() for i in test]
			score={}
			maxm=-100000
			res_class=''
			for c in Class.keys():
				score[c]=math.log2(prior[c])/math.log2(math.exp(1))
				for t in V.keys():
					if t in test:
						score[c]=score[c]+(math.log2(V[t][Class[c]])/math.log2(math.exp(1)))
				if score[c]>maxm:
					maxm=score[c]
					res_class=c
			predictions.append([test_data,test_data[value][0],res_class])

		correct_class=[1 for i in predictions if i[1]==i[2]]
		print('Accuracy of fold ',fold+1,': ',round(len(correct_class)*100/len(test_data),3))
		accuracy=accuracy+len(correct_class)/len(test_data)
	return accuracy/k	

if __name__ == '__main__':
	data=load_dataset("SMSSpamCollection")	
	print('length of dataset: ', len(data))
	for i in range(len(data)):
		first=data[i]
		h=random.randint(0,len(data)-1)
		second=data[h]
		data[i]=deepcopy(second)
		data[h]=deepcopy(first)
	k=5
	accuracy_MV=multivariateNB(deepcopy(data),k)
	print('\nFinal Accuracy of multivariate:', round(accuracy_MV*100,3),'\n')
	accuracy_MN=multinomialNB(deepcopy(data),k)
	print('\nFinal Accuracy of multinomial:', round(accuracy_MN*100,3),'\n')
	t1=time()-t0
	print('time taken :', t1)
	print('\n')
	