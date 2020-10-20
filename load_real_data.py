#!/usr/bin/env
from __future__ import division
import time
import numpy as np
import scipy
from numpy import linalg as LA
#from scipy.sparse import rand, random, find, spdiags, linalg, csr_matrix
from scipy.io import loadmat

#------------------------------------------------------------
# Load various real data matrices from UCI Machine Learning Repository, 
# for SVM related tasks, and get the data into usable format. 

#------------------------------------------------------------
# About the datasets:
#------------------------------------------------------------

# Due to the large size of the datasets, we don't include that datasets 
# in this directory. The user of this code must download the datasets 
# themselves from the UCI Machine Learning Repository.  

# For example, the'ARCENE' dataset is found here: 
# https://archive.ics.uci.edu/ml/datasets/Arcene

# Make sure to modify directory information accordingly, 
# in the calls to 'open' below so that you can open the file. 

#------------------------------------------------------------


def load_ARCENE():
	#f = open('/home/ubuntu/ipm_data/demo/demo_train.data', 'r')
	# f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.data', 'r')
	#f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.data', 'r')
	#f = open('/home/ubuntu/ipm_data/MADELON/madelon_train.data', 'r')

	# TRAIN data
	#f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.data', 'r')
	f = open('/Users/palma/Documents/Work/1_Projects/code_py/ipm/ipm_data/ARCENE/arcene_train.data', 'r')
	
	X_train = readSpaceDelMatrixFile(f)
	f = open('/Users/palma/Documents/Work/1_Projects/code_py/ipm/ipm_data/ARCENE/arcene_train.labels', 'r')
	y_train = readVecFile(f)

	# TEST data
	f = open('/Users/palma/Documents/Work/1_Projects/code_py/ipm/ipm_data/ARCENE/arcene_valid.data', 'r')
	X_test = readSpaceDelMatrixFile(f)
	f = open('/Users/palma/Documents/Work/1_Projects/code_py/ipm/ipm_data/ARCENE/arcene_valid.labels', 'r')
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

def load_gene_exp():
	m = 801
	N = 20531
	f = open('/home/ubuntu/ipm_data/gene/data.csv', 'r')
	X_train = readCSVMatrixFile_geneRNA(f,m,N)
	print X_train[0:8,0:8]

	f = open('/home/ubuntu/ipm_data/gene/labels.csv', 'r')
	y_train = readVecFile_geneRNA(f)
	return X_train,y_train

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

# for gene data, X
def readCSVMatrixFile_geneRNA(f,m,N):
	# file format: 
	# sample_0,0.0,2.01720929003,3.26552691165,5.47848651208...

	# use readline() to read the first line 
	line = f.readline()

	A = np.zeros((m,N))
	row_ind = 0
	while line: #for line in f:
		#print row_ind
		# use realine() to read next line
		line = f.readline()


		line = line.rstrip('\n')

		sVals = line.split(',')    # list of strings 
		sample_num_str = sVals[0]  
		del sVals[0]    # for an array it would be: sVals[1:]   # all but the first element

		#print sVals[1:5]
		fVals = list(map(np.float32, sVals))  # cast the strings as floats

		# check if sVals is not empty
		# had a problem with 1 too many lines taken by "while line:"
		if not sVals:
			break

		#print fVals[1:5]
		A[row_ind,:] = fVals
		#for fVals_k in fVals:
		#		A[row_ind,int(fVals_k)-1] = fVals[] # place in A
		#	#ind = ind + 1

		row_ind = row_ind + 1
	f.close()
	return A

# for gene data, y's
def readVecFile_geneRNA(f):
	# file format: 
	# sample_0,PRAD

	# use readline() to read the first line 
	#line = f.readline()

	# train labels
	gene_list_strings = []
	gene_binary_BRCA = []

	row_ind = 0
	#while True:
	for line in f:
		# use realine() to read next line
		#line = f.readline()

		line = line.rstrip('\n')  

		sVals = line.split(',')   			  # the strings

		sample_num_str = sVals[0]  
		#print sVals
		del sVals[0] # take out "sample_0", to leave a list of strings of  "PRAD" etc
		#print sVals[0]
		gene_list_strings.append(sVals)

		if sVals[0] == 'BRCA':
			gene_binary_BRCA.append(1)
		else:
			gene_binary_BRCA.append(-1)

		# check if line is not empty
		#if not line:
		#	break
	f.close()

	#print gene_binary_BRCA

	y = np.asarray(gene_binary_BRCA, dtype=np.float32)  

	y = np.asarray(y).ravel() # flatten matrix to ndarray !

	#print y
	return y



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

