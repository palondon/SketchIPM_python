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
import load_real_data
from parameters import m,w,n,p,gamma,sigma_step,sigma,tol_cg, MAXIT_cg, DENSITY,power
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# Sketched IPM vs Standard IPM, 1 Run, number of iterations, condition number 
#------------------------------------------------------------

directory = '/Users/palma/Documents/Work/1_Projects/code_py/ipm/ipm_figures'

#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Test IPM SVM real or synthetic dataset, 1 Run', '\n'
	#------------------------------------------------------------


	#------------------------------------------------------------
	# Data LOAD 
	t_data_1 = time.time()
	#X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_1
	X_train, y_train, X_test, y_test, w_true = gen_data.gen_data_SVM(m,n,p,DENSITY)
	

	#X_train,y_train,X_test,y_test = load_real_data.load_ARCENE()
	#X_train,y_train,X_test,y_test = load_real_data.load_DEXTER()
	#X_train,y_train = load_real_data.load_DrivFace() # 606 x 6400
	#X_train,y_train,X_test,y_test = load_real_data.load_DOROTHEA() # (800 x 100k)

	t_data_2 = time.time()
	print 'time data load    = ', t_data_2 - t_data_1, ' secs'

	data_desc = 's' # s for synthetic 
	#data_desc = 'ARCENE'
	print 'data_desc    = ', data_desc


	#------------------------------------------------------------
	# Parameters (for real data)
	# get dimensions 
	m,N = np.shape(X_train)
	#w = 700 # 200 E-6
	#tol_cg = 1E-4
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
	#t_cvx_1 = time.time()
	#x_cvx, p_cvx = svm_func.run_CVXPY_LP(A,b,c)
	#t_cvx_2 = time.time()
	#print'\np* cvx = ', p_cvx
	#print 'time cvx = ', t_cvx_2 - t_cvx_1, ' secs'
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

	#------------------------------------------------------------
	# IPM Ours
	t_ipm_1 = time.time()
	x,y,s,iter_out_ipm,iter_in_cg_vec_ipm,kap_ADW_vec,v_vec,time_ls_ipm = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
	t_ipm_2 = time.time()
	
	p_ipm_final = np.dot(c,x)		
	d_ipm_final = np.dot(b,y)	
	v_vec_mean = np.mean(v_vec)
	print'\nv_vec_mean = ', v_vec_mean
	#------------------------------------------------------------
	print '\n-------------------------- \nSVM Results: \n--------------------------'
	
	# w_approx_ipm = x[0:N] - x[N:2*N]
	# w_approx_cvx = np.transpose(np.squeeze(x_cvx[0:N] - x_cvx[N:2*N]))


	# print 'x = \n',x[0:5]
	# print 'x_lp = \n', x_cvx[0:5]
	# print 'w_approx_ipm = \n',w_approx_ipm[0:5]
	# print 'w_approx_cvx = \n',w_approx_cvx[0:5]

	# diff = np.transpose(x_cvx) - x
	# error_rel = 100*np.linalg.norm(diff,2)/np.linalg.norm(x_cvx,2)
	# print 'error_rel %  = ', error_rel
	
	# print type(X_train)
	# print type(y_train)
	# print np.shape(y_train)
	# train_error_ipm, test_error_ipm = svm_func.train_test_error(X_train,X_test,y_train,y_test,w_approx_ipm)
	# train_error_cvx, test_error_cvx = svm_func.train_test_error(X_train,X_test,y_train,y_test,w_approx_cvx)
	
	# print 'train_error_ipm   = ',train_error_ipm
	# print 'test_error_ipm    = ',test_error_ipm

	# print 'train_error_cvx   = ',train_error_cvx
	# print 'test_error_cvx    = ',test_error_cvx	

	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	print 'time ipm    = ', t_ipm_2 - t_ipm_1, ' secs'
	#print 'time cvx    = ', t_cvx_2 - t_cvx_1, ' secs'

	print '\niter stan out = ', iter_out_stan
	print 'iter ipm out = ', iter_out_ipm

	print '\niter stan in cg = \n', iter_in_cg_vec_stan
	print '\nkap_AD_vec = \n', kap_AD_vec

	print '\niter ipm  in cg = \n', iter_in_cg_vec_ipm
	print '\nkap_ADW_vec = \n', kap_ADW_vec

	print 'p   = ',p_ipm_final
	print 'd   = ',d_ipm_final
	print 'p   = ',p_ipm_vec_stan
	print 'd   = ',d_ipm_vec_stan

	print v_vec
	#------------------------------------------------------------
	# Plot
	msize = 2
	colors = ['o-','ro-','bo-','go-','co-','mo-','ko-','yo-']
	colors_blues = [(0, 0, 1),(0.4, 0.6, 0.9),(0.5, 0.5, 1),(0.11, 0.56, 1),(0.53, 0.80, 0.98)]  # dark to light blues

	#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	#hfont = {'fontname':'Helvetica'}
	plt.rcParams['font.size'] = 16
	smallerfont = 14 # override if needed smaller 
	#------------------------------------------------------------
	# 0 - Outer vs. Inner Iterations  color = (0.5 0.5 1)
	plt.figure(0) 
	plt.semilogy(range(iter_out_stan), iter_in_cg_vec_stan, colors[1], label='Stand. IPM', markersize=8)
	
	#plt.semilogy(range(iter_out_stan), iter_in_cg_vec_stan, colors[0], label='Stand. IPM', markersize=8)
	plt.semilogy(range(iter_out_ipm), iter_in_cg_vec_ipm, colors[2], label='Sketch IPM', markersize=8)
	
	plt.legend(loc='upper left') # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel('Inner CG Iterations')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	#plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_0.eps', format='eps', dpi=1200)
	plt.savefig(directory +'/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1run_e'+str(power)+'_0.eps', format='eps', dpi=1200)
	
	#------------------------------------------------------------
	# 1 - Outer vs. Inner Iterations  color = (0.5 0.5 1)
	plt.figure(1) 
	plt.plot(range(iter_out_ipm), iter_in_cg_vec_ipm, colors[2], label='Sketch IPM', markersize=8)
	
	plt.legend(loc='upper left') # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel('Inner CG Iterations')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	#plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_0.eps', format='eps', dpi=1200)
	plt.savefig(directory +'/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1run_e'+str(power)+'_1.eps', format='eps', dpi=1200)
	
	#------------------------------------------------------------
	# 2 - kappa
	plt.figure(2)
	plt.semilogy(range(iter_out_stan), kap_AD_vec, colors[1], label=r'$ \kappa(A D^2 A^T)$ Stand. IPM', markersize=8)
	plt.semilogy(range(iter_out_ipm), kap_ADW_vec, colors[2], label=r'$\kappa(Q^{-1}A D^2 A^T)$ Sketch IPM', markersize=8)
	
	#ax.set_yscale('log')
	plt.legend(loc='upper left') # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel('Condition Number')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	#plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_1.eps', format='eps', dpi=1200)
	plt.savefig(directory +'/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1run_e'+str(power)+'_2.eps', format='eps', dpi=1200)
	
	
	#------------------------------------------------------------
	# 3 - kappa just sketched IPM
	plt.figure(3)
	plt.plot(range(iter_out_ipm), kap_ADW_vec, colors[2], label=r'$\kappa(Q^{-1}A D^2 A^T)$ Sketch IPM', markersize=8)
	
	#ax.set_yscale('log')
	plt.legend(loc='upper left') # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel('Condition Number')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	#plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_1.eps', format='eps', dpi=1200)
	plt.savefig(directory +'/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1run_e'+str(power)+'_3.eps', format='eps', dpi=1200)
	

	#------------------------------------------------------------
	# 4 - v
	plt.figure(4)
	plt.semilogy(range(iter_out_ipm), v_vec, colors[2], markersize=8)
	
	#ax.set_yscale('log')
	#plt.legend(loc='upper left') # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel(r'$\|\|v\|\|_2$')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	#plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_2.eps', format='eps', dpi=1200)
	plt.savefig(directory +'/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1run_e'+str(power)+'_4.eps', format='eps', dpi=1200)
	
	

