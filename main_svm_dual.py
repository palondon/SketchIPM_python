#!/usr/bin/env
#------------------------------------------------------------
import cvxpy as cvx
#import multiprocessing as mp
#from multiprocessing import Pool
#import string
import time
import numpy as np
import scipy as sp
import pandas as pd
import csv

from scipy import optimize
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
#------------------------------------------------------------
import gen_data 	
import ipm_func_real
from parameters import p,gamma,sigma_step,sigma,t_iter, MAXIT_cg,tol_cg, DENSITY
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# IPM SVM real data
#------------------------------------------------------------


#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Test IPM MADELON dataset', '\n'
	#------------------------------------------------------------


	#------------------------------------------------------------
	# Data 
	t_data_1 = time.time()
	#X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)

	#f = open('/home/ubuntu/ipm_data/ARCENE/temp.data', 'r')

	# READ the data 
	resultList = []
	#f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.data', 'r')
	#f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.data', 'r')
	f = open('/home/ubuntu/ipm_data/MADELON/madelon_train.data', 'r')
	for line in f:
		line = line.rstrip(' \n')  # SUPER IMPORTANT: need the SPACE here b/c the file has them
		sVals = line.split(' ')   				# the strings
		fVals = list(map(np.float32, sVals))  	# the floats
		resultList.append(fVals)  # put the new row in the ndarray
	f.close()
	X_train = np.asarray(resultList, dtype=np.float32)  

	# READ the labels
	resultList2 = []
 	#f = open('/home/ubuntu/ipm_data/ARCENE/arcene_train.labels', 'r')
 	#f = open('/home/ubuntu/ipm_data/DEXTER/dexter_train.labels', 'r')
 	f = open('/home/ubuntu/ipm_data/MADELON/madelon_train.labels', 'r')
	for line in f:
		line = line.rstrip(' \n')  # NO SPACE here b/c the label file is different! 
		sVals = line.split(' ')   			  # the strings
		fVals = list(map(np.float32, sVals))  # the floats
		resultList2.append(fVals)  # put the new row in the ndarray
	f.close()
	y_train = np.asarray(resultList2, dtype=np.float32)  
	# this is not a csv (Comma separated file)
	# Although it was named after comma-separated values, the CSV module can manage parsed files regardless 
	# of the field delimiter - be it tabs
	#print type(df)

	print np.shape(X_train)
	print type(X_train)
	print np.shape(y_train)
	print type(y_train)

	#------------------------------------------------------------
	# Parameters (for real data)
	# get dimensions 
	m,N = np.shape(X_train)
	w = 700
	n = 2*N+1

	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print 'p = ', p
	print 'w = ', w
	print '\n', '---------------------------'


	#------------------------------------------------------------
	# form l1-SVM constraint matrix 
	yX = y_train*X_train
	#A = np.concatenate((-yX, yX, -1*y_train),axis=1)
	#b = -1*np.ones(m)
	#c = np.concatenate((np.ones(2*N),np.array([0])))
	
	t_data_2 = time.time()
	print 'time data generate    = ', t_data_2 - t_data_1, ' secs'
	#------------------------------------------------------------
	# form the dual
	AA = np.concatenate((-yX, yX, -1*y_train),axis=1)
	bb = -1*np.ones(m)
	cc = np.concatenate((np.ones(2*N),np.array([0])))
	
	A = np.transpose(AA)
	b = cc
	c = bb
	N,m = np.shape(X_train)
	w = 700
	n = 2*N+1

	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print 'p = ', p
	print 'w = ', w
	print '\n', '---------------------------'

	#------------------------------------------------------------
	# CVX LP
	x = cvx.Variable(n)
	objective1 = cvx.Minimize(c*x) # c*log(x)
	constraints1 = [A*x == b, 0 <= x] # , 1 >= x
	prob = cvx.Problem(objective1, constraints1)

	t_cvx_1 = time.time()
	prob.solve() # SCS ECOS solver=cvx.SCS
	t_cvx_2 = time.time()
	print'\np* cvx = ', prob.value
	print 'time cvx = ', t_cvx_2 - t_cvx_1, ' secs'
	#------------------------------------------------------------
	# IPM Ours
	t_ipm_1 = time.time()
	x,y,s = ipm_func_real.ipm_real(m,n,w,A,b,c)
	t_ipm_2 = time.time()
	
	p_ipm_vec = np.dot(c,x)		
	d_ipm_vec = np.dot(b,y)	
	#print 'x = \n',x
	#print 'x_lp = \n', res.x

	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	print 'time ipm    = ', t_ipm_2 - t_ipm_1, ' secs'
	print 'time cvx    = ', t_cvx_2 - t_cvx_1, ' secs'

	print 'p   = ',p_ipm_vec
	print 'd   = ',d_ipm_vec
	print 'cvx = ', prob.value

