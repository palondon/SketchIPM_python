#!/usr/bin/env
#------------------------------------------------------------
import cvxpy as cvx
#import multiprocessing as mp
#from multiprocessing import Pool
#import string
import time
import numpy as np
import scipy as sp
from scipy import optimize
from scipy.io import loadmat
#from scipy.sparse import rand, find, spdiags, linalg, csr_matrix
import matplotlib
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
#------------------------------------------------------------
import gen_data 	
import ipm_func
import svm_func
from parameters import m,n,p,w,N,gamma,sigma_step,sigma,t_iter, MAXIT_cg,tol_cg, DENSITY
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# IPM main SVM simulated data -- 1 Run
#------------------------------------------------------------


#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Test IPM', '\n'
	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print 'p = ', p
	print 'w = ', w
	print '\n', '---------------------------'
	#------------------------------------------------------------


	#------------------------------------------------------------
	# Data 
	t_data_1 = time.time()
	X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)

	# form l1-SVM constraint matrix 
	# multiply rows of X by y_train_i
	yX = np.transpose(np.multiply(np.transpose(X_train), y_train))
	A = np.concatenate((-yX, yX, -1*y_train[:, None]),axis=1) # axis with the different dim
	b = -1*np.ones(m)
	c = np.concatenate((np.ones(2*N),np.array([0]))) # note: (w^+) + (w^-) (not a -)

	t_data_2 = time.time()
	print 'time data generate    = ', t_data_2 - t_data_1, ' secs'
	#------------------------------------------------------------
	# CVX LP
	t_cvx_1 = time.time()
	x_cvx, p_cvx = svm_func.run_CVXPY_LP(A,b,c)
	t_cvx_2 = time.time()

	print'\np* cvx = ', p_cvx
	print 'time cvx = ', t_cvx_2 - t_cvx_1, ' secs'

	#------------------------------------------------------------
	# IPM Ours
	t_ipm_1 = time.time()
	x,y,s,t_iter_ipm,t_ls_ipm,iter_cg_ipm,v_vec = ipm_func.ipm(m,n,w,A,b,c)
	t_ipm_2 = time.time()
	
	p_ipm_vec = np.dot(c,x)		
	d_ipm_vec = np.dot(b,y)	

	#------------------------------------------------------------
	print '\n-------------------------- \nSVM Results: \n--------------------------'
	
	w_approx_ipm = x[0:N] - x[N:2*N]
	w_approx_cvx = np.transpose(np.squeeze(x_cvx[0:N] - x_cvx[N:2*N]))


	print 'x = \n',x[0:5]
	print 'x_lp = \n', x_cvx[0:5]
	print 'w_approx_ipm = \n',w_approx_ipm[0:5]
	print 'w_approx_cvx = \n',w_approx_cvx[0:5]

	diff = np.transpose(x_cvx) - x
	error_rel = 100*np.linalg.norm(diff,1)/np.linalg.norm(x_cvx,1)
	print 'error_rel %  = ', error_rel
	
	print type(X_train)
	print type(y_train)
	print 'np.shape(y_train)   = ', np.shape(y_train)

	train_error_ipm, test_error_ipm = svm_func.train_test_error(X_train,X_test,y_train,y_test,w_approx_ipm)
	train_error_cvx, test_error_cvx = svm_func.train_test_error(X_train,X_test,y_train,y_test,w_approx_cvx)
	
	print 'train_error_ipm   = ',train_error_ipm
	print 'test_error_ipm    = ',test_error_ipm

	print 'train_error_cvx   = ',train_error_cvx
	print 'test_error_cvx    = ',test_error_cvx	

	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	print 'time ipm    = ', t_ipm_2 - t_ipm_1, ' secs'
	print 'time cvx    = ', t_cvx_2 - t_cvx_1, ' secs'

	print 'p   = ',p_ipm_vec
	print 'd   = ',d_ipm_vec
	print 'cvx = ', p_cvx


