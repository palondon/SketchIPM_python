#!/usr/bin/env
#------------------------------------------------------------
import time
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
#------------------------------------------------------------
import gen_data 	
import ipm_func
import svm_func
import sim_func
from parameters import m,n,p,gamma,sigma_step,sigma, MAXIT_cg, DENSITY,directory
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# Heatmap (w vs CG_tol) vs Iterations, Condition number 
#------------------------------------------------------------
# For 1 synthetic data set:
# loop over w and CG_tol --> to make a grid/heatmap plot

# main_w_tolcg_grid
#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Heatmap of (w vs CG_tol) vs Iterations, and Condition number', '\n'

	# Figure 2 (a) heatmap 
	# Figure 2 (b)  

	data_desc = 's' # s for synthetic 
	#data_desc = 'ARCENE'
	#data_desc = 'DEXTER'
	#data_desc = 'DrivFace'
	#data_desc = 'DOROTHEA'

	# Loop over w and tol_cg;
	# pick several dataset-appropriate sketching dimensions  
	w_vec = [20,40,60]  # example; for a (m x n) = (10 x 100) data set 
	tol_cg_vec = [1E-5,1E-3,1E-2]
	#tol_cg_vec = [3E-6,3E-5]
	# tol_cg_vec = [3E-6,6E-6,1E-5,3E-5]

	#------------------------------------------------------------
	# Data LOAD 
	t_data_1 = time.time()
	if data_desc == 's': # s for synthetic 
		X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
	elif data_desc == 'ARCENE':
		X_train,y_train,X_test,y_test = load_real_data.load_ARCENE()
	elif data_desc == 'DEXTER':
		X_train,y_train,X_test,y_test = load_real_data.load_DEXTER()
	elif data_desc == 'DrivFace':
		X_train,y_train = load_real_data.load_DrivFace() # 606 x 6400
	elif data_desc == 'DOROTHEA':
		X_train,y_train = load_real_data.load_DrivFace() # (800 x 100k)
	t_data_2 = time.time()	

	print 'data_desc    = ', data_desc
	print 'time data load    = ', t_data_2 - t_data_1, ' secs'

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

	# ARCENE (100 x 10K) good range:
	#w_vec = [200, 300, 500, 1000] 
	#tol_cg_vec = [3E-6,6E-6,1E-5,3E-5]

	#w_vec = [200, 300, 500, 700, 1000] 
	#tol_cg_vec = [3E-6,6E-6,9E-6,1E-5,3E-5]

	#w_vec = [200, 300, 500, 700, 1000, 1500, 2000] 
	#tol_cg_vec = [1E-6,2E-6,3E-6,5E-6,6E-6,8E-6,9E-6,1E-5]

	#w_vec = [100, 500, 700, 1000] 
	#tol_cg_vec = [1E-6,5E-6,8E-6,1E-5] # [1E-6,5E-6,1E-5]

	#------------------------------------------------------------
	# loop 
	sim_func.loop_w_tolcg(X_train,y_train,w_vec,tol_cg_vec,data_desc)
	#------------------------------------------------------------


