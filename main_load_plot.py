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
from parameters import m,n,p,gamma,sigma_step,sigma,t_iter, MAXIT_cg, DENSITY
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
	print 'Load saved outputs and plot', '\n'
	print '---------------------------', '\n'
	#------------------------------------------------------------
	# get the information for the dataset you want to plot
	X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
	#X_train,y_train,X_test,y_test = svm_func.load_ARCENE()

	data_desc = 's' # s for synthetic 
	#data_desc = 'ARCENE'
	print 'data_desc    = ', data_desc

	# Parameters
	# get dimensions 
	m,N = np.shape(X_train)
	n = 2*N+1
	#------------------------------------------------------------
	w_vec = np.loadtxt('/home/ubuntu/ipm_out/w_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	tol_cg_vec = np.loadtxt('/home/ubuntu/ipm_out/tol_cg_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	
	v_norm = np.loadtxt('/home/ubuntu/ipm_out/v_norm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	it_ipm = np.loadtxt('/home/ubuntu/ipm_out/it_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	it_sta = np.loadtxt('/home/ubuntu/ipm_out/it_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	er_rel = np.loadtxt('/home/ubuntu/ipm_out/er_rel_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	

	print w_vec
	print tol_cg_vec

	print '\nv_norm = \n', v_norm
	print 'iter ipm = \n', it_ipm
	print 'iter sta = \n', it_sta
	print 'error rel= \n', er_rel
	#num_w = len(w_vec)

	#num_tol_cg = len(tol_cg_vec)
	
	#------------------------------------------------------------
	# take a subset of the data points computed
	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	
	ind_w_subset = range(num_w)
	ind_tol_subset = range(num_tol_cg)

	# ARCENE
	#ind_w_subset = [0,1,2,4,5]
	#ind_tol_subset = [0,2,4,5,7]


	#------------------------------------------------------------
	# Plot (the second time, if want to correct the 1st plots formating, etc)
	sim_func.plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,er_rel,data_desc,ind_w_subset,ind_tol_subset)



