#!/usr/bin/env
#------------------------------------------------------------
import time
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import rc
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
#------------------------------------------------------------
import gen_data 	
import ipm_func
import svm_func
import load_real_data
#import sim_func
import plot_func
from parameters import m,n,p,gamma,sigma_step,sigma, MAXIT_cg, DENSITY,directory
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# Sketched IPM vs Standard IPM

# -- 1 dataset: pick one of {'ARCENE','DEXTER','DrivFace','DOROTHEA'  or  synthetic}
# -- multiple w's (sketching dimension) 
# -- plot and save the data at the end 

# Produces Figure 1

#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Run Sketched IPM on real or synthetic dataset SVM, 1 Run', '\n'
	#------------------------------------------------------------

	# Figure 1 (a) Outer Iteration vs Inner Iteration (CG or PCG iterations) and 
	# Figure 1 (b) Outer Iteration vs Condition number 

	# pick a data set (uncomment) and choose several sketching dimensions to try
 	data_desc = 's' # s for synthetic 
	#data_desc = 'ARCENE'
	#data_desc = 'DEXTER'
	#data_desc = 'DrivFace'
	#data_desc = 'DOROTHEA'

	# pick several dataset-appropriate sketching dimensions 
	w_vec = [200,400,600] # synethic (10 x 100) matrix 
	#w_vec = [20,40,100] # good for 'ARCENE'


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
	# Parameters (for real data cases)
	# get dimensions 
	m,N = np.shape(X_train)
	tol_cg = 1E-5#1E-6 # change later? 
	n = 2*N+1
	#------------------------------------------------------------
	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print 'w = ', w_vec
	print 'tol_cg = ', tol_cg
	print '\n', '---------------------------'
	#------------------------------------------------------------
	# form l1-SVM constraint matrix 
	A,b,c = svm_func.formL1SVM(X_train,y_train)
	#------------------------------------------------------------
	# IPM Standard
	t_stan_1 = time.time()
	x,y,s,iter_out_stan,iter_in_cg_vec_stan,kap_AD_vec,time_ls_stan = ipm_func.ipm_standard(m,n,A,b,c,tol_cg)
	t_stan_2 = time.time()
	
	p_ipm_vec_stan = np.dot(c,x)		
	d_ipm_vec_stan = np.dot(b,y)
	print '\niter stan out = ', iter_out_stan
	
	print 'p   = ',p_ipm_vec_stan
	print 'd   = ',d_ipm_vec_stan

	print 'test   = ', iter_out_stan
	#------------------------------------------------------------
	# IPM Sketch (Ours)
	w = w_vec[0]
	x,y,s,iter_out_ipm_2,iter_in_cg_vec_ipm_2,kap_ADW_vec_2,v_vec_2,time_ls_ipm = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
	
	w = w_vec[1]
	x,y,s,iter_out_ipm_5,iter_in_cg_vec_ipm_5,kap_ADW_vec_5,v_vec_5,time_ls_ipm = ipm_func.ipm(m,n,w,A,b,c,tol_cg)

	w = w_vec[2]
	x,y,s,iter_out_ipm_8,iter_in_cg_vec_ipm_8,kap_ADW_vec_8,v_vec_8,time_ls_ipm = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
	
	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	#print 'time ipm    = ', t_ipm_2 - t_ipm_1, ' secs'

	#print '\niter stan out = ', iter_out_stan
	#print 'iter ipm out = ', iter_out_ipm

	#print '\niter stan in cg = \n', iter_in_cg_vec_stan
	#print '\nkap_AD_vec = \n', kap_AD_vec

	#print '\niter ipm  in cg = \n', iter_in_cg_vec_ipm
	#print '\nkap_ADW_vec = \n', kap_ADW_vec

	p_ipm_final = np.dot(c,x)		
	d_ipm_final = np.dot(b,y)	

	print 'p   = ',p_ipm_final
	print 'd   = ',d_ipm_final
	print 'p   = ',p_ipm_vec_stan
	print 'd   = ',d_ipm_vec_stan

	print '\niter stan out = ', iter_out_stan

	# plot right away
	plot_func.plot_it_mul_w(m,n,data_desc, \
		iter_out_stan,iter_in_cg_vec_stan,iter_out_ipm_2,iter_in_cg_vec_ipm_2,iter_out_ipm_5, \
		iter_in_cg_vec_ipm_5,iter_out_ipm_8,iter_in_cg_vec_ipm_8, \
		kap_AD_vec,kap_ADW_vec_2,kap_ADW_vec_5,kap_ADW_vec_8,v_vec_2,v_vec_5,v_vec_8)


	# also save output, if want to re-plot later
	np.savetxt(directory +'ipm_out/iter_out_stan_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', np.array(iter_out_stan).reshape(1,), fmt='%i') # int
	np.savetxt(directory +'ipm_out/iter_in_cg_vec_stan_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', iter_in_cg_vec_stan, fmt='%i') # int

	np.savetxt(directory +'ipm_out/iter_out_ipm_2_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', np.array(iter_out_ipm_2).reshape(1,), fmt='%i')
	np.savetxt(directory +'ipm_out/iter_in_cg_vec_ipm_2_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', iter_in_cg_vec_ipm_2, fmt='%i') # int
	
	np.savetxt(directory +'ipm_out/iter_out_ipm_5_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', np.array(iter_out_ipm_5).reshape(1,), fmt='%i')
	np.savetxt(directory +'ipm_out/iter_in_cg_vec_ipm_5_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', iter_in_cg_vec_ipm_5, fmt='%i') # int
	
	np.savetxt(directory +'ipm_out/iter_out_ipm_8_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', np.array(iter_out_ipm_8).reshape(1,), fmt='%i')
	np.savetxt(directory +'ipm_out/iter_in_cg_vec_ipm_8_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', iter_in_cg_vec_ipm_8, fmt='%i') # int
	
	np.savetxt(directory +'ipm_out/kap_AD_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', kap_AD_vec, fmt='%.10f')
	np.savetxt(directory +'ipm_out/kap_ADW_vec_2_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', kap_ADW_vec_2, fmt='%.10f')
	np.savetxt(directory +'ipm_out/kap_ADW_vec_5_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', kap_ADW_vec_5, fmt='%.10f')
	np.savetxt(directory +'ipm_out/kap_ADW_vec_8_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', kap_ADW_vec_8, fmt='%.10f')

	np.savetxt(directory +'ipm_out/v_vec_2_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', v_vec_2, fmt='%.10f')
	np.savetxt(directory +'ipm_out/v_vec_5_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', v_vec_5, fmt='%.10f')
	np.savetxt(directory +'ipm_out/v_vec_8_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', v_vec_8, fmt='%.10f')

