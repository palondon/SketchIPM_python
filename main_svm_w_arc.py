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
from parameters import gamma,sigma_step,sigma,t_iter, MAXIT_cg, DENSITY
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
	print 'Test SVM ARCENE real dataset', '\n'
	#------------------------------------------------------------
	# Data LOAD
	t_data_1 = time.time()
	#X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
	X_train,y_train,X_test,y_test = svm_func.load_ARCENE()
	#X_train,y_train,X_test,y_test = svm_func.load_DEXTER()

	#data_desc = 's' # s for synthetic 
	data_desc = 'ARCENE'
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

	# Arcene: 100 x 10K
	w_vec = [200, 300, 500, 700, 1000, 1500, 2000] 
	#tol_cg_vec = [1E-6,5E-6,8E-6,1E-5] # OG
	tol_cg_vec = [1E-6,2E-6,3E-6,5E-6,6E-6,8E-6,9E-6,1E-5]
	
	#w_vec = [100, 500, 700, 1000] 
	#tol_cg_vec = [1E-6,5E-6,8E-6,1E-5] # [1E-6,5E-6,1E-5]

	#------------------------------------------------------------
	# loop 
	sim_func.loop_w_tolcg(X_train,y_train,w_vec,tol_cg_vec,data_desc)
	#------------------------------------------------------------


