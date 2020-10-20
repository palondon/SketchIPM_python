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
#import sim_func
import plot_func
from parameters import m,n,p,gamma,sigma_step,sigma, MAXIT_cg, DENSITY,directory
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
	w_vec = np.loadtxt(directory+'ipm_out/w_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	tol_cg_vec = np.loadtxt(directory+'ipm_out/tol_cg_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	
	v_norm = np.loadtxt(directory+'ipm_out/v_norm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')

	it_ipm = np.loadtxt(directory+'ipm_out/it_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	it_sta = np.loadtxt(directory+'ipm_out/it_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	
	it_cg_ipm = np.loadtxt(directory+'ipm_out/it_cg_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	it_cg_sta = np.loadtxt(directory+'ipm_out/it_cg_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	
	kap_ipm = np.loadtxt(directory+'ipm_out/kap_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	kap_sta = np.loadtxt(directory+'ipm_out/kap_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')

	#er_rel = np.loadtxt(directory+'ipm_out/er_rel_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt')
	

	print 'w_vec        = ',w_vec
	print 'tol_cg_vec   = ',tol_cg_vec

	print '\nv_norm = \n', v_norm
	print 'iter ipm = \n', it_ipm
	print 'iter sta = \n', it_sta

	print 'it_cg_ipm = \n', it_cg_ipm
	print 'it_cg_sta = \n', it_cg_sta

	print 'kap_ipm = \n', kap_ipm
	print 'kap_sta = \n', kap_sta

	#print 'error rel= \n', er_rel
	#num_w = len(w_vec)

	#num_tol_cg = len(tol_cg_vec)
	
	#------------------------------------------------------------
	# take a subset of the data points computed
	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	
	ind_w_subset = range(num_w)
	ind_tol_subset = range(num_tol_cg)

	# ARCENE
	#ind_w_subset = [0,1,2,3,4]
	#ind_tol_subset = [0,2,4,5,7]


	#------------------------------------------------------------
	# Plot (the second time, if want to correct the 1st plots formating, etc)
	#plot_func.plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,it_cg_ipm,it_cg_sta,kap_ipm,kap_sta,er_rel,data_desc,ind_w_subset,ind_tol_subset)
	plot_func.plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,it_cg_ipm,it_cg_sta,kap_ipm,kap_sta,data_desc,ind_w_subset,ind_tol_subset)


