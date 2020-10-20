#!/usr/bin/env
#------------------------------------------------------------
import time
import numpy as np
import scipy as sp
import csv

import matplotlib
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
from matplotlib import rc
#------------------------------------------------------------
import gen_data 	
import ipm_func
import svm_func
import plot_func
from parameters import gamma,sigma_step,sigma, MAXIT_cg, DENSITY,directory
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# functions related to doing the simulation - running over loops of w, tol_cg, etc


def loop_w_tolcg(X_train,y_train,w_vec,tol_cg_vec,data_desc):
	# loop over various (w, tol_cg) pairs.
	# save the results as text files, then plot. (can be re-ploted later)
	#------------------------------------------------------------
	t_total_1 = time.time()
	#------------------------------------------------------------
	# form l1-SVM constraint matrix 
	A,b,c = svm_func.formL1SVM(X_train,y_train)
	#------------------------------------------------------------
	# Parameters
	# get dimensions 
	m,N = np.shape(X_train)
	n = 2*N+1
	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print '\n', '---------------------------'
	#------------------------------------------------------------
	# CVX LP
	#t_cvx_1 = time.time()
	#x_cvx, p_cvx = svm_func.run_CVXPY_LP(A,b,c)
	#t_cvx_2 = time.time()
	#print'\np* cvx = ', p_cvx
	#print 'time cvx = ', t_cvx_2 - t_cvx_1, ' secs'
	#------------------------------------------------------------
	# Loop over w and tol_cg

	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	v_norm = np.zeros((num_w,num_tol_cg))	
	it_ipm = np.zeros((num_w,num_tol_cg))	 	# outer iterations  ours
	it_sta = np.zeros(num_tol_cg) 				# outer iterations  standard

	it_cg_ipm = np.zeros((num_w,num_tol_cg))	 # inner CG iterations  ours
	it_cg_sta = np.zeros(num_tol_cg) 			 # inner CG iterations  standard

	kap_ipm = np.zeros((num_w,num_tol_cg))	 # condi #
	kap_sta = np.zeros(num_tol_cg) 			 # condi #

	#er_rel = np.zeros((num_w,num_tol_cg))	

	#------------------------------------------------------------
	for ind_tol_cg in range(num_tol_cg):
		tol_cg = tol_cg_vec[ind_tol_cg]
		#------------------------------------------------------------
		# IPM Standard
		t_ls_stan = 0
		x,y,s,t_iter_stan,iter_in_cg_stan_vec,kap_AD_vec,time_ls = ipm_func.ipm_standard(m,n,A,b,c,tol_cg)
		#x,y,s,t_iter_stan,t_ls_stan = ipm_func.ipm_standard(m,n,A,b,c,0)
		#it_sta = t_iter_stan*np.ones((num_w,num_tol_cg))	
		
		it_sta[ind_tol_cg] = t_iter_stan
		it_cg_sta[ind_tol_cg] = np.ndarray.max(iter_in_cg_stan_vec)
		kap_sta[ind_tol_cg] = np.ndarray.max(kap_AD_vec)

		p_ipm_vec_stan = np.dot(c,x)		
		d_ipm_vec_stan = np.dot(b,y)

		print 'p stan = ',p_ipm_vec_stan
		print 'd stan = ',d_ipm_vec_stan
		#------------------------------------------------------------
		for ind_w in range(num_w):
			w = w_vec[ind_w]
			#------------------------------------------------------------
			# IPM Ours
			x,y,s,iter_out_ipm,iter_in_cg_ipm_vec,kap_ADW_vec,v_vec,time_ls = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
			#x,y,s,t_iter_ipm,t_ls,iter_cg_ipm,v_vec = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
			
			p_ipm_final = np.dot(c,x)		
			d_ipm_final = np.dot(b,y)	
			print 'p ipm  = ',p_ipm_final
			print 'd ipm  = ',d_ipm_final
			v_vec_mean = np.mean(v_vec)
			#------------------------------------------------------------
			# post process
			#diff = np.transpose(x_cvx) - x
			#error_rel = 100*np.linalg.norm(diff,2)/np.linalg.norm(x_cvx,2)

			#------------------------------------------------------------
			# Record
			v_norm[ind_w,ind_tol_cg] = v_vec_mean
			it_ipm[ind_w,ind_tol_cg] = iter_out_ipm
			it_cg_ipm[ind_w,ind_tol_cg] = np.ndarray.max(iter_in_cg_ipm_vec)
			kap_ipm[ind_w,ind_tol_cg] = np.ndarray.max(kap_ADW_vec)
			#er_rel[ind_w,ind_tol_cg] = error_rel
			#------------------------------------------------------------
		#------------------------------------------------------------
	#------------------------------------------------------------



	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	#print 'time ipm    = ', t_ipm_2 - t_ipm_1, ' secs'
	#print 'time cvx    = ', t_cvx_2 - t_cvx_1, ' secs'

	print 'w_vec        = ',w_vec
	print 'tol_cg_vec   = ',tol_cg_vec

	print '\nv_norm = \n', v_norm
	print 'iter ipm = \n', it_ipm
	print 'iter sta = \n', it_sta

	print 'it_cg_ipm = \n', it_cg_ipm
	print 'it_cg_sta = \n', it_cg_sta

	print 'kap_ipm = \n', kap_ipm
	print 'kap_sta = \n', kap_sta
	

	t_total_2 = time.time()
	print 'time total    = ', t_total_2 - t_total_1, ' secs'
	#------------------------------------------------------------
	np.savetxt(directory+'ipm_out/w_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', w_vec) # int
	np.savetxt(directory+'ipm_out/w_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', w_vec, fmt='%i') # int
	np.savetxt(directory+'ipm_out/tol_cg_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', tol_cg_vec, fmt='%.10f')

	np.savetxt(directory+'ipm_out/v_norm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', v_norm, fmt='%.10f')
	np.savetxt(directory+'ipm_out/it_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', it_ipm, fmt='%i') # int
	np.savetxt(directory+'ipm_out/it_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', it_sta, fmt='%i') # int
	
	np.savetxt(directory+'ipm_out/it_cg_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', it_cg_ipm, fmt='%i') # int
	np.savetxt(directory+'ipm_out/it_cg_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', it_cg_sta, fmt='%i') # int
	
	np.savetxt(directory+'ipm_out/kap_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', kap_ipm, fmt='%i') # int
	np.savetxt(directory+'ipm_out/kap_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', kap_sta, fmt='%i') # int
	
	#np.savetxt(directory+'ipm_out/er_rel_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', er_rel, fmt='%.10f')
	
	# take all the data; not subset of the data (might want to take a subset later)
	#ind_w_subset = range(num_w)
	#ind_tol_subset = range(num_tol_cg)

	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	
	ind_w_subset = range(num_w)
	ind_tol_subset = range(num_tol_cg)

	#------------------------------------------------------------
	# Plot: call main_load_plot.py

	# could have plotted directly after running the algorithm: 
	# Plot (the first time) Can plot again using main_plot if want to change something 
	# plot_func.plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,it_cg_ipm,it_cg_sta,kap_ipm,kap_sta,data_desc,ind_w_subset,ind_tol_subset)
	
	#------------------------------------------------------------
