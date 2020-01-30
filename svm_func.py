#!/usr/bin/env
from __future__ import division
import time
import numpy as np
import scipy
from numpy import linalg as LA
from scipy.sparse import rand,random, find, spdiags, linalg, csr_matrix
from scipy.io import loadmat
#from cvxpy import *
import cvxpy as cvx
#------------------------------------------------------------
# from sklearn.datasets import fetch_rcv1
# import sklearn
#------------------------------------------------------------
#import matplotlib
#matplotlib.use('Agg') # needed for using matplotlib on remote machine
#import matplotlib.pyplot as plt
#------------------------------------------------------------
# functions for SVM related tasks
#------------------------------------------------------------

def train_test_error(X_train,X_test,y_train,y_test,w_approx):
	m_train ,_= X_train.shape
	m_test ,_= X_test.shape
	
	print np.shape(np.sign(y_train ))
	print np.shape(np.sign(np.dot(X_train,w_approx) ))
	train_error =  100*(np.sign(y_train ) != np.sign(np.dot(X_train,w_approx) )).sum()/m_train 
	test_error  =  100*(np.sign(y_test )  != np.sign(np.dot(X_test,w_approx) )).sum()/m_test

	return train_error, test_error 

def formL1SVM(X_train,y_train):
	#------------------------------------------------------------
	# form l1-SVM constraint matrix 
	yX = np.transpose(np.multiply(np.transpose(X_train), y_train))
	A = np.concatenate((-yX, yX, -1*y_train[:, None]),axis=1) # axis with the different dim
	
	m,N = np.shape(X_train)

	#yX = y_train*X_train
	#A = np.concatenate((-yX, yX, -1*y_train),axis=1)
	b = -1*np.ones(m)
	c = np.concatenate((np.ones(2*N),np.array([0])))

	return A,b,c



def run_CVXPY_LP(A,b,c):
	#------------------------------------------------------------
	# CVX LP
	m,n = A.shape

	x_cvx = cvx.Variable(n)
	objective1 = cvx.Minimize(c*x_cvx) # c*log(x)
	constraints1 = [A*x_cvx == b, 0 <= x_cvx] # , 1 >= x
	prob = cvx.Problem(objective1, constraints1)
	prob.solve() # SCS ECOS solver=cvx.SCS

	x_cvx_out = np.asarray(x_cvx.value).ravel() # flatten matrix to ndarray !

	return x_cvx_out, prob.value


def load_ARCENE():
	#f = open('/home/ubuntu/ipm_data/demo/demo_train.data', 'r')
	# f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.data', 'r')
	#f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.data', 'r')
	#f = open('/home/ubuntu/ipm_data/MADELON/madelon_train.data', 'r')

	# TRAIN data
	f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.data', 'r')
	X_train = readSpaceDelMatrixFile(f)
	f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.labels', 'r')
	y_train = readVecFile(f)

	# TEST data
	f = open('/home/ubuntu/ipm_data/ARCENE/arcene_valid.data', 'r')
	X_test = readSpaceDelMatrixFile(f)
	f = open('/home/ubuntu/ipm_data/ARCENE/arcene_valid.labels', 'r')
	y_test = readVecFile(f)

	return X_train,y_train,X_test,y_test


def load_DEXTER():
	m = 300
	m_test = 300
	N = 20000

	# TRAIN data
	f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.data', 'r')
	X_train = readColonDelMatrixFile(f,m,N)

	# train labels
 	f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.labels', 'r')
	y_train = readVecFile(f)

	# TEST data
	f = open('/home/ubuntu/ipm_data/DEXTER/dexter_valid.data', 'r')
	X_test = readColonDelMatrixFile(f,m_test,N)

	# test labels
 	f = open('/home/ubuntu/ipm_data/DEXTER/dexter_valid.labels', 'r')
	y_test = readVecFile(f) 

	print 'X_train = ',np.shape(X_train)
	#print 'X_train = ',type(X_train)
	print 'y_train = ',np.shape(y_train)
	#print 'y_train = ',type(y_train)

	print 'X_test = ',np.shape(X_test)
	#print 'X_test = ',type(X_test)
	print 'y_test = ',np.shape(y_test)
	#print 'y_test = ',type(y_test)

	return X_train,y_train,X_test,y_test

def load_DOROTHEA():
	m = 800
	m_test = 350
	N = 100000

	f = open('/home/ubuntu/ipm_data/DOROTHEA/dorothea_train.data', 'r')
	X_train = readSpaceDelIndexMatrixFile(f,m,N)

	f = open('/home/ubuntu/ipm_data/DOROTHEA/dorothea_train.labels', 'r')
	y_train = readVecFile(f)

	f = open('/home/ubuntu/ipm_data/DOROTHEA/dorothea_valid.data', 'r')
	X_test = readSpaceDelIndexMatrixFile(f,m_test,N)

	f = open('/home/ubuntu/ipm_data/DOROTHEA/dorothea_valid.labels', 'r')
	y_test = readVecFile(f) 

	return X_train,y_train,X_test,y_test


def load_DrivFace():

	# returns a dict
	DrivFace_dict = loadmat('/home/ubuntu/ipm_data/DrivFace/DrivFace.mat')
	drivFaceD = DrivFace_dict['drivFaceD'] # use the struct name as in Matlab

	# in Matlab, drivFaceD is the 'struct', which has 'fields'
	# in Matlab, access the data by: 
	# drivFaceD.data  # training dataset 
	# drivFaceD.nlab  # labels
	# drivFaceD.lablist  #(1,2,3 categories)
	
	#print drivFaceD.dtype # print the list of fields of the struct
	#print drivFaceD[0, 0]['lablist']
	X_train = drivFaceD[0, 0]['data']
	y_train_123 = drivFaceD[0, 0]['nlab']

	y_train_123 = np.asarray(y_train_123).ravel() # flatten to (1,) ndarray !


	print type(X_train)
	print np.shape(X_train)
	print type(y_train_123)
	print np.shape(y_train_123)
	#print y_train_123

	# for binary classification: 

	# 1's and 3's become 1 group: "looking to the left or right side group" --> {+1} group
	# 2's are other group: "looking straight ahead"  --> {-1} group
	#y_train_123[y_train_123 == 3] = 1
	#y_train_123[y_train_123 == 2] = -1
	y_train = np.where(y_train_123==3, 1, y_train_123) 
	y_train = np.where(y_train==2, -1, y_train) 
	#print y_train

	return X_train,y_train


def readSpaceDelMatrixFile(f):
	# Read a text file with a matrix stored in simple form: 
	# 'value value value ' with SPACE @ end of ea. line
	# f: file
	resultList = []
	for line in f:
		line = line.rstrip(' \n')  # need the SPACE here b/c file has a space at the end of each row.
		sVals = line.split(' ')   				# the numbers, read as strings; "what's in between each space"
		fVals = list(map(np.float32, sVals))  	# cast the strings as floats
		resultList.append(fVals)  # put the new row in the ndarray
	f.close()
	A = np.asarray(resultList, dtype=np.float32)  
	return A


def readVecFile(f):
	# Read a text file with a vector stored in a list, each entry on new line, no spaces
	# f: file
	# train labels
	resultList1 = []
	for line in f:
		line = line.rstrip(' \n')  # NO SPACE here b/c the label file is different! 
		sVals = line.split(' ')   			  # the strings
		fVals = list(map(np.float32, sVals))  # the floats
		resultList1.append(fVals)  # put the new row in the ndarray
	f.close()
	y = np.asarray(resultList1, dtype=np.float32)  

	y = np.asarray(y).ravel() # flatten matrix to ndarray !

	return y

def readSpaceDelIndexMatrixFile(f,m,N):
	# Read a text file with a "sparse binary matrix" stored in form:
	# 'ind ind ind ' where all the ind correspond to a "1" in the matrix, 0's are elsewere
	# f: file
	# note: ind read in are "starting at 1"
	A = np.zeros((m,N))
	row_ind = 0
	for line in f:
		line = line.rstrip(' \n')  # need the SPACE here b/c file has a space at the end of each row.
		sVals = line.split(' ')    # list of strings 
		fVals = list(map(np.float32, sVals))  # cast the strings as floats
		#ind = 0
		for fVals_k in fVals:
			A[row_ind,int(fVals_k)-1] = 1 # place in A
			#ind = ind + 1
		row_ind = row_ind + 1
	f.close()
	return A


def readColonDelMatrixFile(f,m,N):
	# Read a text file with a matrix stored in form often used for sparse matrix (ind, value) pair:
	# 'ind:value'... 'ind:value' with SPACE @ end of ea. line
	# f: file
	# (m,N): size of matrix to be read (change later to generalize)
	A = np.zeros((m,N))
	#f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.data', 'r')
	row_ind = 0
	for line in f:
		line = line.rstrip(' \n')  # need the SPACE here b/c file has a space at the end of each row.
		sVals = line.split(' ')    # list of strings of what's inbetween each space:  'ind:num'
		for sVals_k in sVals:
			indAndNum_strings = sVals_k.split(':')  # split the ind:num at the : colon
			indAndNum_floats = list(map(np.float32, indAndNum_strings))  # cast the strings as floats
			#print indAndNum_floats
			A[row_ind,int(indAndNum_floats[0])] = indAndNum_floats[1] # place in A
		row_ind = row_ind + 1
	f.close()
	return A

