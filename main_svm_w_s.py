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
import sim_func
from parameters import m,n,N,p,gamma,sigma_step,sigma,t_iter, MAXIT_cg, DENSITY
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# IPM SVM real data
#------------------------------------------------------------
# For 1 data set, loop over w and CG_tol

#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Test SVM synthetic dataset', '\n'
	#------------------------------------------------------------
	# Data LOAD
	t_data_1 = time.time()
	X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
	#X_train,y_train,X_test,y_test = svm_func.load_ARCENE()

	data_desc = 's' # s for synthetic 
	#data_desc = 'ARCENE'
	print 'data_desc    = ', data_desc

	t_data_2 = time.time()
	print 'time data gen    = ', t_data_2 - t_data_1, ' secs'
	#------------------------------------------------------------
	# Parameters
	# get dimensions 
	m,N = np.shape(X_train)
	n = 2*N+1
	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print '\n', '---------------------------'

	#------------------------------------------------------------
	# Loop over w and tol_cg

	# 100 x 1000
	#w_vec = [200,400,500,700] 
	#tol_cg_vec = [1E-6,2E-6,5E-6,8E-6,1E-5]
	#tol_cg_vec = [1E-5,1.2E-4,1.5E-4,2E-4]  $ too large 

	# 50 x 100
	# w_vec = [50,51,52,55,60] 
	#tol_cg_vec = [1E-5,1.5E-4,1.8E-4,2E-4,2.2E-4] 

	w_vec = [50,51,52] 
	tol_cg_vec = [1E-5,1.5E-4,1.8E-4] 

	#------------------------------------------------------------
	# loop 
	sim_func.loop_w_tolcg(X_train,y_train,w_vec,tol_cg_vec,data_desc)
	#------------------------------------------------------------


