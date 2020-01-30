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
from parameters import gamma,sigma_step,sigma,t_iter, MAXIT_cg, DENSITY
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# functions related to doing the simulation - running over loops of w, tol_cg, etc


def loop_w_tolcg(X_train,y_train,w_vec,tol_cg_vec,data_desc):
	# loop, save, plot
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
	t_cvx_1 = time.time()
	x_cvx, p_cvx = svm_func.run_CVXPY_LP(A,b,c)
	t_cvx_2 = time.time()
	print'\np* cvx = ', p_cvx
	print 'time cvx = ', t_cvx_2 - t_cvx_1, ' secs'
	#------------------------------------------------------------
	# Loop over w and tol_cg

	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	v_norm = np.zeros((num_w,num_tol_cg))	
	it_ipm = np.zeros((num_w,num_tol_cg))	
	#it_sta = np.zeros((num_w,num_tol_cg))	
	er_rel = np.zeros((num_w,num_tol_cg))	

	#------------------------------------------------------------
	# IPM Standard
	t_ls_stan = 0
	x,y,s,t_iter_stan,t_ls_stan = ipm_func.ipm_standard(m,n,A,b,c,0)
	it_sta = t_iter_stan*np.ones((num_w,num_tol_cg))	
	
	p_ipm_vec_stan = np.dot(c,x)		
	d_ipm_vec_stan = np.dot(b,y)
	print 'p stan = ',p_ipm_vec_stan
	print 'd stan = ',d_ipm_vec_stan
	#------------------------------------------------------------
	for ind_w in range(num_w):
		w = w_vec[ind_w]
		for ind_tol_cg in range(num_tol_cg):
			tol_cg = tol_cg_vec[ind_tol_cg]
			#------------------------------------------------------------
			# IPM Ours
			x,y,s,t_iter_ipm,t_ls,iter_cg_ipm,v_vec = ipm_func.ipm(m,n,w,A,b,c,tol_cg)
			
			p_ipm_final = np.dot(c,x)		
			d_ipm_final = np.dot(b,y)	
			print 'p ipm  = ',p_ipm_final
			print 'd ipm  = ',d_ipm_final
			v_vec_mean = np.mean(v_vec)
			#------------------------------------------------------------
			# post process
			diff = np.transpose(x_cvx) - x
			error_rel = 100*np.linalg.norm(diff,2)/np.linalg.norm(x_cvx,2)

			#------------------------------------------------------------
			# Record
			v_norm[ind_w,ind_tol_cg] = v_vec_mean
			it_ipm[ind_w,ind_tol_cg] = t_iter_ipm
			#it_sta[ind_w,ind_tol_cg] = t_iter_stan
			er_rel[ind_w,ind_tol_cg] = error_rel
			#------------------------------------------------------------
		#------------------------------------------------------------
	#------------------------------------------------------------



	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	#print 'time ipm    = ', t_ipm_2 - t_ipm_1, ' secs'
	#print 'time cvx    = ', t_cvx_2 - t_cvx_1, ' secs'

	print '\nv_norm = \n', v_norm
	print 'iter ipm = \n', it_ipm
	print 'iter sta = \n', it_sta
	print 'error rel= \n', er_rel

	t_total_2 = time.time()
	print 'time total    = ', t_total_2 - t_total_1, ' secs'
	#------------------------------------------------------------
	np.savetxt('/home/ubuntu/ipm_out/w_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', w_vec) # int
	np.savetxt('/home/ubuntu/ipm_out/w_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', w_vec, fmt='%i') # int
	np.savetxt('/home/ubuntu/ipm_out/tol_cg_vec_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', tol_cg_vec, fmt='%.10f')

	np.savetxt('/home/ubuntu/ipm_out/v_norm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', v_norm, fmt='%.10f')
	np.savetxt('/home/ubuntu/ipm_out/it_ipm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', it_ipm, fmt='%i') # int
	np.savetxt('/home/ubuntu/ipm_out/it_sta_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', it_sta, fmt='%i') # int
	np.savetxt('/home/ubuntu/ipm_out/er_rel_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'.txt', er_rel, fmt='%.10f')
	
	# take all the data; not subset of the data (might want to take a subset later)
	ind_w_subset = range(num_w)
	ind_tol_subset = range(num_tol_cg)
	#------------------------------------------------------------
	# Plot (the first time) Can plot again using main_plot if want to change something 
	#plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,er_rel,data_desc,ind_w_subset,ind_tol_subset)
	#------------------------------------------------------------



def plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,er_rel,data_desc,ind_w_subset,ind_tol_subset):
	#------------------------------------------------------------
	# Plot
	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	#------------------------------------------------------------

	#w_vec = w_vec.astype(int) # w_vec was saved as floats

	#------------------------------------------------------------

	# w_vec = [100,200]
	# tol_cg_vec
	msize = 2
	colors = ['o-','ro-','bo-','go-','co-','mo-','ko-','yo-']

	#hfont = {'fontname':'Helvetica'}
	plt.rcParams['font.size'] = 16
	smallerfont = 14 # override if needed smaller 
	#plt.rcParams['font.family'] = 'Helvetica' # change later for formatting safety

	#plt.title('(m x n) = ('+str(m)+" x "+str(n)+"), t = "+str(num_t)+", s = "+str(numbersubprob))
	#plt.gcf().subplots_adjust(bottom=0.16) # increase moves up
	#plt.gcf().subplots_adjust(left=0.18) # increase moves to the right

	#------------------------------------------------------------
	# 0 - w vs. v_norm
	plt.figure(0)
	#for ind_tol in range(num_tol_cg):
	for ind_tol in ind_tol_subset:
		plt.plot(w_vec, v_norm[:,ind_tol], colors[ind_tol], label=r'CG tol $ = '+str(tol_cg_vec[ind_tol])+'$', markersize=8)
	
	plt.legend(loc='upper right') # , fontsize = 18
	plt.xlabel(r'Sketch Dim. $w$')
	plt.ylabel(r'$\|v\|_2$')  # ,format='%.e'
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_0.eps', format='eps', dpi=1200)
	#------------------------------------------------------------
	# 1 - w vs. it_ipm
	plt.figure(1)
	#for ind_tol in range(num_tol_cg):
	for ind_tol in ind_tol_subset:
		plt.plot(w_vec, it_ipm[:,ind_tol], colors[ind_tol], label=r'CG tol $ = '+str(tol_cg_vec[ind_tol])+'$', markersize=8)
	
	plt.legend(loc='upper right')
	plt.xlabel(r'Sketch Dim. $w$')
	plt.ylabel('Outer Iterations') 
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1.eps', format='eps', dpi=1200)


	#------------------------------------------------------------
	# 2-  tol_cg vs. v_norm
	plt.figure(2)
	#for ind_w2 in range(num_w):
	for ind_w2 in ind_w_subset:
		plt.plot(tol_cg_vec, v_norm[ind_w2,:], colors[ind_w2], label=r'$w = '+str(w_vec[ind_w2])+'$', markersize=8)
	
	plt.xscale("log")
	plt.legend(loc='upper left')
	plt.xlabel('Rel. Tolerance CG')
	plt.ylabel(r'$|v|_2$')  # ,format='%.e'
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_2.eps', format='eps', dpi=1200)
	#------------------------------------------------------------
	# 3- tol_cg vs. it_ipm
	plt.figure(3)
	#for ind_w2 in range(num_w):
	for ind_w2 in ind_w_subset:
		plt.plot(tol_cg_vec, it_ipm[ind_w2,:], colors[ind_w2], label=r'$w = '+str(w_vec[ind_w2])+'$', markersize=8)
	
	plt.xscale("log")
	plt.legend(loc='upper left')
	plt.xlabel('Rel. Tolerance CG')
	plt.ylabel('Outer Iterations') 
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_3.eps', format='eps', dpi=1200)


	#------------------------------------------------------------
	# for the heatmaps, need to reshape, 
	# if taking a subset of the recorded information
	w_vec = w_vec[ind_w_subset]
	tol_cg_vec = tol_cg_vec[ind_tol_subset]
	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	v_norm = v_norm[np.ix_(ind_w_subset, ind_tol_subset)]
	it_ipm = it_ipm[np.ix_(ind_w_subset, ind_tol_subset)]

	#------------------------------------------------------------
	# heatmaps
	plt.figure(4)
	plt.imshow(v_norm, cmap=plt.cm.Blues);
	
	#plt.imshow(v_norm, cmap=plt.cm.Blues, interpolation='none', extent=[1,num_tol_cg,w_vec[0],w_vec[num_w-1]]);
	# tol_cg_vec[0],tol_cg_vec[num_tol_cg-1]

	# w labels 
	yy = w_vec.astype(int) # w_vec was saved as floats
	ny = num_w
	no_labels = num_w # how many labels to see on axis x
	step_y = int(ny / (no_labels - 1)) # step between consecutive labels
	y_positions = np.arange(0,ny,step_y) # pixel count at label position
	y_labels = yy[::step_y] # labels you want to see
	
	#x_labels_int = [int(i) for i in x_labels] 
	plt.yticks(y_positions, y_labels)

	# tol_cg labels 
	xx = tol_cg_vec
	nx = num_tol_cg
	no_labels = num_tol_cg # how many labels to see on axis x
	step_x = int(nx / (no_labels - 1)) # step between consecutive labels
	x_positions = np.arange(0,nx,step_x) # pixel count at label position
	x_labels = xx[::step_x] # labels you want to see
	
	#print np.format_float_scientific(np.float32(np.pi))
	#x_labels = np.format_float_scientific(x_labels, exp_digits=1)
	plt.xticks(x_positions, x_labels, fontsize = smallerfont)

	plt.colorbar(format='%.0e') # pl.colorbar(myplot, format='%.0e')
	plt.xlabel('Rel. Tolerance CG') 
	plt.ylabel(r'Sketch Dim. $w$')
	plt.title(r'$|v|_2$')
	
	#plt.show()
	plt.tight_layout()
	plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_4_im_v.pdf', format='pdf')


	#------------------------------------------------------------
	plt.figure(5)
	plt.imshow(it_ipm, cmap=plt.cm.Blues);
	plt.yticks(y_positions, y_labels) # (from above) 
	plt.xticks(x_positions, x_labels, fontsize = smallerfont) # (from above)
	plt.colorbar()
	plt.xlabel('Rel. Tolerance CG') 
	plt.ylabel(r'Sketch Dim. $w$')
	plt.title('Outer Iterations')
	#plt.show()
	plt.tight_layout()
	plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_5_im_iter.pdf', format='pdf')

	#------------------------------------------------------------













