#!/usr/bin/env
from __future__ import division
import time
import numpy as np
import scipy
from numpy import linalg as LA
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

