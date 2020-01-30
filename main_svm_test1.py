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
import ipm_func
import svm_func
from parameters import m,n,p,gamma,sigma_step,sigma,t_iter, MAXIT_cg, DENSITY
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
	print 'Test IPM SVM real or synthetic dataset, 1 Run', '\n'
	#------------------------------------------------------------


	#------------------------------------------------------------
	# Data LOAD 
	t_data_1 = time.time()
	#X_train,y_train,X_test,y_test = svm_func.load_ARCENE()
	#X_train,y_train,X_test,y_test = svm_func.load_DEXTER()

	X_train,y_train = svm_func.load_DrivFace() # 606 x 6400

	#X_train,y_train,X_test,y_test = svm_func.load_DOROTHEA() # (800 x 100k)
	#X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
	t_data_2 = time.time()
	print 'time data load    = ', t_data_2 - t_data_1, ' secs'

	#------------------------------------------------------------
	# Parameters (for real data)
	# get dimensions 
	m,N = np.shape(X_train)
	w = 5000
	tol_cg = 1E-8
	n = 2*N+1
	#------------------------------------------------------------
	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print 'w = ', w
	print 'tol_cg = ', tol_cg
	print '\n', '---------------------------'
	#------------------------------------------------------------
	# form l1-SVM constraint matrix 
	A,b,c = svm_func.formL1SVM(X_train,y_train)
	#------------------------------------------------------------
	# CVX LP
	t_cvx_1 = time.time()
	x_cvx, p_cvx = svm_func.run_CVXPY_LP(A,b,c)
	t_cvx_2 = time.time()
	print'\np* cvx = ', p_cvx
	print 'time cvx = ', t_cvx_2 - t_cvx_1, ' secs'
	#------------------------------------------------------------
	# IPM Standard
	t_stan_1 = time.time()
	x,y,s,iter_stan,t_ls_stan = ipm_func.ipm_standard(m,n,A,b,c,tol_cg)
	t_stan_2 = time.time()
	
	p_ipm_vec_stan = np.dot(c,x)		
	d_ipm_vec_stan = np.dot(b,y)
	print '\niter stan = ', iter_stan
	
	print 'p   = ',p_ipm_vec_stan
	print 'd   = ',d_ipm_vec_stan

	#------------------------------------------------------------
	# IPM Ours
	t_ipm_1 = time.time()
	x,y,s,iter_ipm,t_ls,iter_cg_ipm,v_vec = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
	t_ipm_2 = time.time()
	
	p_ipm_final = np.dot(c,x)		
	d_ipm_final = np.dot(b,y)	
	v_vec_mean = np.mean(v_vec)
	print'\nv_vec_mean = ', v_vec_mean
	#------------------------------------------------------------
	print '\n-------------------------- \nSVM Results: \n--------------------------'
	
	w_approx_ipm = x[0:N] - x[N:2*N]
	w_approx_cvx = np.transpose(np.squeeze(x_cvx[0:N] - x_cvx[N:2*N]))


	print 'x = \n',x[0:5]
	print 'x_lp = \n', x_cvx[0:5]
	print 'w_approx_ipm = \n',w_approx_ipm[0:5]
	print 'w_approx_cvx = \n',w_approx_cvx[0:5]

	diff = np.transpose(x_cvx) - x
	error_rel = 100*np.linalg.norm(diff,2)/np.linalg.norm(x_cvx,2)
	print 'error_rel %  = ', error_rel
	
	print type(X_train)
	print type(y_train)
	print np.shape(y_train)
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

	print '\niter stan = ', iter_stan
	print 'iter ipm  = ', iter_ipm

	print 'p   = ',p_ipm_final
	print 'd   = ',d_ipm_final
	print 'p   = ',p_ipm_vec_stan
	print 'd   = ',d_ipm_vec_stan
	print 'cvx = ', p_cvx


