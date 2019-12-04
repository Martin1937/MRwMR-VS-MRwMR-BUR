import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

import numpy as np
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split

from mdentropy import mutinf,cmutinf
import tensorflow as tf
import tensorflow.contrib.layers as layers

import time
import math
import scipy.stats
from scipy.stats import entropy

#Load Data
data = pd.DataFrame()
data = pd.read_csv("Data.csv",index_col = None,header=None)
data = data.values
label = data[:,-1]
data = data[:,0:-1]


#Define the classifier
MLP = MLPClassifier(alpha=1, hidden_layer_sizes=(20,15,7,3),max_iter=1000,learning_rate_init=0.01)
SVM = svm.SVC(kernel='linear')
RandomForest = RandomForestClassifier(n_estimators=50, criterion='entropy')

beta = 0.1
num_of_features = 150

#Function to calculate the UR of each feature
#Input: 2D Array:(Sample, Features) and Label
#Return: UR of each single features
def bias_value(input_data_x,input_data_y):
	bias_num = []
	input_data_y = np.reshape(input_data_y,(-1,1))
	input_data_x = np.array(input_data_x)
	input_data_x_old = input_data_x
	high_score = mutinf(100,input_data_x,input_data_y)
	input_data_x_old = input_data_x
	for i in range(0,input_data_x.shape[1]):
		input_data_x = np.delete(input_data_x,i,1)
		diff = high_score - mutinf(100,input_data_x,input_data_y)
		bias_num.append(diff)
		input_data_x = input_data_x_old
	bias_num = np.array(bias_num)
	bias_num[bias_num<0] = 0
	return bias_num

#All Scoring Functions below take features and label as input
#return the indexes of features in a descending order of all features' scores

#MIM Scoring Function
def MIM_selection(temp_data,temp_label):
	MI = mutual_info_classif(temp_data,temp_label)
	selection_order = np.argsort(MI)[::-1]
	return selection_order

#MIM Scoring Function with BUR
def MIM_selection_UR(temp_data,temp_label):
	bias = bias_value(temp_data,temp_label)
	MI = mutual_info_classif(temp_data,temp_label)
	MI_UR = (1-beta)*MI + bias*beta 
	selection_order_BUR = np.argsort(MI_UR)[::-1]
	return selection_order_BUR

#JMI Scoring Function
def JMI_selection(input_data_x,input_data_y):
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	select_index = []
	select_index.append(max_mi)
	selected_features = input_data_x[:,max_mi].reshape(-1,1)
	diff = []
	for x in range(0,num_of_features):
		for i in range(0,input_data_x.shape[1]):
			key = 0
			for j in range(0,len(select_index)):
				if i == select_index[j]:
					key = 1
			if key == 0:
				feature = input_data_x[:,i].reshape(-1,1)
				MI = 0
				for k in range(0,selected_features.shape[1]):
					MI = MI + mutinf(100,feature,np.reshape(selected_features[:,k],(-1,1)))
				diff.append(MI)
			else:
				diff.append(-99.9)
		max_new = np.argmax(diff)
		select_index.append(max_new)
		del diff[:]
		selected_features = np.append(selected_features,input_data_x[:,max_new].reshape(-1,1),axis=1)
	return select_index

#JMI Scoring Function with BUR
def JMI_selection_UR(input_data_x,input_data_y):
	bias = bias_value(input_data_x,input_data_y)
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	select_index = []
	select_index.append(max_mi)
	selected_features = input_data_x[:,max_mi].reshape(-1,1)
	diff = []
	for x in range(0,num_of_features):
		for i in range(0,input_data_x.shape[1]):
			key = 0
			for j in range(0,len(select_index)):
				if i == select_index[j]:
					key = 1
			if key == 0:
				feature = input_data_x[:,i].reshape(-1,1)
				MI = 0
				for k in range(0,selected_features.shape[1]):
					MI = MI + mutinf(100,feature,np.reshape(selected_features[:,k],(-1,1)))
				diff.append((1-beta)*MI/selected_features.shape[1]+beta*bias[i])
			else:
				diff.append(-99.9)
		max_new = np.argmax(diff)
		select_index.append(max_new)
		del diff[:]
		selected_features = np.append(selected_features,input_data_x[:,max_new].reshape(-1,1),axis=1)
	return select_index

#mRMR Scoring Function
def mrmr_selection(input_data_x,input_data_y):
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	select_index = []
	select_index.append(max_mi)
	selected_features = input_data_x[:,max_mi].reshape(-1,1)
	diff = []
	for x in range(0,num_of_features):
		#print(str(x)+' / '+str(num_of_features))
		for i in range(0,input_data_x.shape[1]):
			key = 0
			for j in range(0,len(select_index)):
				if i == select_index[j]:
					key = 1
			if key == 0:
				feature = input_data_x[:,i]
				MI = 0
				count = 0
				for k in range(0,selected_features.shape[1]):
					MI = MI + mutinf(100,feature.reshape(-1,1),np.reshape(selected_features[:,k],(-1,1)))
					count = count + 1
				diff.append(mutinf(100,feature.reshape(-1,1),input_data_y) - float(MI)/count)
			else:
				diff.append(-99.9)
		max_new = np.argmax(diff)
		select_index.append(max_new)
		del diff[:]
		selected_features = np.append(selected_features,input_data_x[:,max_new].reshape(-1,1),axis=1)
	return select_index

#mRMR Scoring Function with BUR
def mrmr_selection_UR(input_data_x,input_data_y):
	bias = bias_value(input_data_x,input_data_y)
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	select_index = []
	select_index.append(max_mi)
	selected_features = input_data_x[:,max_mi].reshape(-1,1)
	diff = []
	for x in range(0,num_of_features):
		for i in range(0,input_data_x.shape[1]):
			key = 0
			for j in range(0,len(select_index)):
				if i == select_index[j]:
					key = 1
			if key == 0:
				feature = input_data_x[:,i]
				MI = 0
				count = 0
				for k in range(0,selected_features.shape[1]):
					MI = MI + mutinf(100,feature.reshape(-1,1),np.reshape(selected_features[:,k],(-1,1)))
					count = count + 1
				diff.append((1-beta)*(mutinf(100,feature.reshape(-1,1),input_data_y) - float(MI)/count)+beta*bias[i])
			else:
				diff.append(-99.9)
		max_new = np.argmax(diff)
		select_index.append(max_new)
		del diff[:]
		selected_features = np.append(selected_features,input_data_x[:,max_new].reshape(-1,1),axis=1)
	return select_index

#JMIM Scoring Function
def JMIM_selection(input_data_x,input_data_y):
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	select_index = []
	select_index.append(max_mi)
	selected_features = input_data_x[:,max_mi].reshape(-1,1)
	for x in range(0,num_of_features):
		all_min = []
		for i in range(0,input_data_x.shape[1]):
			key = 0
			for j in range(0,len(select_index)):
				if i == select_index[j]:
					key = 1
			if key == 0:
				feature = input_data_x[:,i]
				MI = []
				for k in range(0,selected_features.shape[1]):
					MI.append(mutinf(100,np.reshape(feature,(-1,1)),np.reshape(selected_features[:,k],(-1,1))))
				all_min.append(np.min(MI))
			else:
				all_min.append(-99.9)
		max_new = np.argmax(all_min)
		select_index.append(max_new)
		selected_features = np.append(selected_features,input_data_x[:,max_new].reshape(-1,1),axis=1)
	return select_index

#JMIM Scoring Function with BUR
def JMIM_selection_UR(input_data_x,input_data_y):
	bias = bias_value(input_data_x,input_data_y)
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	select_index = []
	select_index.append(max_mi)
	selected_features = input_data_x[:,max_mi].reshape(-1,1)
	for x in range(0,num_of_features):
		all_min = []
		for i in range(0,input_data_x.shape[1]):
			key = 0
			for j in range(0,len(select_index)):
				if i == select_index[j]:
					key = 1
			if key == 0:
				feature = input_data_x[:,i]
				MI = []
				for k in range(0,selected_features.shape[1]):
					MI.append(mutinf(100,np.reshape(feature,(-1,1)),np.reshape(selected_features[:,k],(-1,1))))
				all_min.append((1-beta)*np.min(MI)+beta*bias[i])
			else:
				all_min.append(-99.9)
		max_new = np.argmax(all_min)
		select_index.append(max_new)
		selected_features = np.append(selected_features,input_data_x[:,max_new].reshape(-1,1),axis=1)
	return select_index

#GSA Scoring Function
def GSA_selection(input_data_x,input_data_y):
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	data_new = input_data_x[:,max_mi]
	all_data = np.reshape(data_new,(-1,1))
	select_index = []
	select_index.append(max_mi)
	for i in range(0,num_of_features):
		score = []
		all_data_old = all_data
		for j in range(0,input_data_x.shape[1]):
			key = 1
			for k in range(0,len(select_index)):
				if j == select_index[k]:
					key = 0
			if key == 1:
				all_data = np.append(all_data,np.reshape(input_data_x[:,j],(-1,1)),axis=1)
				score.append(mutinf(100,np.reshape(all_data,(-1,i+2)),input_data_y))
			else:
				score.append(-99.9)
			all_data = all_data_old

		new_index = np.argmax(score)
		select_index.append(new_index)
		all_data = np.append(all_data,np.reshape(input_data_x[:,new_index],(-1,1)),axis=1)	
	return select_index

#GSA Scoring Function with BUR
def GSA_selection_UR(input_data_x,input_data_y):
	bias = bias_value(input_data_x,input_data_y)
	max_mi = np.argmax(mutual_info_classif(input_data_x,input_data_y))
	input_data_y = np.reshape(input_data_y,(-1,1))
	data_new = input_data_x[:,max_mi]
	all_data = np.reshape(data_new,(-1,1))
	select_index = []
	select_index.append(max_mi)
	for i in range(0,num_of_features):
		score = []
		all_data_old = all_data
		for j in range(0,input_data_x.shape[1]):
			key = 1
			for k in range(0,len(select_index)):
				if j == select_index[k]:
					key = 0
			if key == 1:
				all_data = np.append(all_data,np.reshape(input_data_x[:,j],(-1,1)),axis=1)
				score.append(mutinf(100,np.reshape(all_data,(-1,i+2)),input_data_y)*(1-beta) + beta * bias[j])
			else:
				score.append(-99.9)
			all_data = all_data_old
		new_index = np.argmax(score)
		select_index.append(new_index)
		all_data = np.append(all_data,np.reshape(input_data_x[:,new_index],(-1,1)),axis=1)	
	return select_index

#Take order of features, trainint data, training label, test data and test label
#Use selected features to train the classifier and report the test data accuracy 
def Calculation(order,train_data,train_label,vali_data,vali_label):
	feature_train = []
	feature_vali = []
	MLP_acc = []
	SVM_acc = []
	RF_acc = []
	for index in range(0,num_of_features):
		if len(feature_train)==0:
			feature_train = train_data[:,order[index]].reshape(-1,1)
			feature_vali = vali_data[:,order[index]].reshape(-1,1)
		else:
			feature_train = np.hstack((feature_train,train_data[:,order[index]].reshape(-1,1)))
			feature_vali = np.hstack((feature_vali,vali_data[:,order[index]].reshape(-1,1)))
		
		MLP.fit(feature_train,train_label)
		pred_result = MLP.predict(feature_vali)
		MLP_acc.append(metrics.accuracy_score(vali_label,pred_result))

		SVM.fit(feature_train,train_label)
		pred_result = SVM.predict(feature_vali)
		SVM_acc.append(metrics.accuracy_score(vali_label,pred_result))

		RandomForest.fit(feature_train,train_label)
		pred_result = RandomForest.predict(feature_vali)
		RF_acc.append(metrics.accuracy_score(vali_label,pred_result))

	return MLP_acc,SVM_acc,RF_acc

#MIM, JMI, MRMR, JMIM

method = ['MIM_selection','MIM_selection_UR','mrmr_selection','mrmr_selection_UR','JMI_selection',
'JMI_selection_UR','JMIM_selection','JMIM_selection_UR','GSA_selection','GSA_selection_UR']

dispatcher = {'MIM_selection':MIM_selection,   'MIM_selection_UR': MIM_selection_UR,
			'mrmr_selection':mrmr_selection,   'mrmr_selection_UR':mrmr_selection_UR,
			'JMI_selection':JMI_selection,     'JMI_selection_UR':JMI_selection_UR,
			'JMIM_selection':JMIM_selection,   'JMIM_selection_UR':JMIM_selection_UR,
			'GSA_selection':GSA_selection,     'GSA_selection_UR':GSA_selection_UR}


#Run 10 trails
for i in range(0,10):
	print("Iteration "+str(i))
	x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.25)
	for iter in range(0,10):
		print("Current Method: "+str(method[iter]))
		order_gen = dispatcher[method[iter]](x_train,y_train)
		MLP_result,SVM_result,RF_result = Calculation(order_gen,x_train,y_train,x_test,y_test)

		print("---MLP RESULT---")
		print(MLP_result)
		print("---SVM RESULT---")
		print(SVM_result)
		print("---RF RESULT---")
		print(RF_result)
