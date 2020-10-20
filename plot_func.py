#!/usr/bin/env
#------------------------------------------------------------
#import cvxpy as cvx
#import multiprocessing as mp
#from multiprocessing import Pool
#import string
import time
import numpy as np
import scipy as sp
#import pandas as pd
import csv

#from scipy import optimize
#from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
from matplotlib import rc
#------------------------------------------------------------
import gen_data 	
import ipm_func
import svm_func
from parameters import gamma,sigma_step,sigma, MAXIT_cg, DENSITY, directory
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# load and plot subroutines to produce the final plots
#------------------------------------------------------------

def plot_w_tolcg(m,n,w_vec,tol_cg_vec,v_norm,it_ipm,it_sta,it_cg_ipm,it_cg_sta,kap_ipm,kap_sta,data_desc,ind_w_subset,ind_tol_subset):
	#------------------------------------------------------------
	# Plots for the heat maps (Figure 2)
	#------------------------------------------------------------
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
	rc('font',**{'family':'serif','serif':['Times']})

	plt.rcParams['font.size'] = 20
	smallerfont = 17 # override if needed smaller 
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
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_0.eps', format='eps', dpi=1200)
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
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_1.eps', format='eps', dpi=1200)


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
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_2.eps', format='eps', dpi=1200)
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
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_3.eps', format='eps', dpi=1200)

	#------------------------------------------------------------
	# Inner - iteration plots
	#------------------------------------------------------------
	# 0b w vs. kap_ipm
	plt.figure(4)
	#for ind_tol in range(num_tol_cg):
	for ind_tol in ind_tol_subset:
		plt.plot(w_vec, kap_ipm[:,ind_tol], colors[ind_tol], label=r'CG tol $ = '+str(tol_cg_vec[ind_tol])+'$', markersize=8)
	
	plt.legend(loc='upper right', fontsize = smallerfont) # , fontsize = 18
	plt.xlabel(r'Sketch Dim. $w$')
	plt.ylabel(r'$\kappa(Q^{-1/2}A D^2 A^T Q^{-1/2})$')  # ,format='%.e'
	plt.yscale("log")

	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_4.eps', format='eps', dpi=1200)
	
	#------------------------------------------------------------
	# 1b - w vs. it_cg_ipm
	plt.figure(5)
	#for ind_tol in range(num_tol_cg):
	for ind_tol in ind_tol_subset:
		plt.plot(w_vec, it_cg_ipm[:,ind_tol], colors[ind_tol], label=r'CG tol $ = '+str(tol_cg_vec[ind_tol])+'$', markersize=8)
	
	plt.legend(loc='upper right', fontsize = smallerfont)
	plt.xlabel(r'Sketch Dim. $w$')
	plt.ylabel('Inner Iterations') 
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_5.eps', format='eps', dpi=1200)

	#------------------------------------------------------------
	# 2b -  tol_cg vs. kappa
	plt.figure(6)
	#for ind_w2 in range(num_w):
	for ind_w2 in ind_w_subset:
		plt.plot(tol_cg_vec, kap_ipm[ind_w2,:], colors[ind_w2], label=r'$w = '+str(w_vec[ind_w2])+'$', markersize=8)
	
	plt.xscale("log")
	plt.legend(loc='upper left', fontsize = smallerfont)
	plt.xlabel('Rel. Tolerance CG')
	plt.ylabel(r'$\kappa$')  # ,format='%.e'
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_6.eps', format='eps', dpi=1200)
	

	#------------------------------------------------------------
	# 3b - tol_cg vs. inner it
	plt.figure(7)
	#for ind_w2 in range(num_w):
	for ind_w2 in ind_w_subset:
		plt.plot(tol_cg_vec, it_cg_ipm[ind_w2,:], colors[ind_w2], label=r'$w = '+str(w_vec[ind_w2])+'$', markersize=8)
	
	#plt.xscale("log")
	plt.legend(loc='upper left', fontsize = smallerfont)
	plt.xlabel('Rel. Tolerance CG')
	plt.ylabel('Inner Iterations') 
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_7.eps', format='eps', dpi=1200)





	#------------------------------------------------------------
	# for the heatmaps, need to reshape, 
	# if taking a subset of the recorded information
	w_vec = w_vec[ind_w_subset]
	tol_cg_vec = tol_cg_vec[ind_tol_subset]
	num_w = len(w_vec)
	num_tol_cg = len(tol_cg_vec)
	v_norm = v_norm[np.ix_(ind_w_subset, ind_tol_subset)]
	it_ipm = it_ipm[np.ix_(ind_w_subset, ind_tol_subset)]

	it_cg_ipm = it_cg_ipm[np.ix_(ind_w_subset, ind_tol_subset)]
	kap_ipm = kap_ipm[np.ix_(ind_w_subset, ind_tol_subset)]


	#------------------------------------------------------------
	# heatmaps 
	
	# x and y axis for all the heatmap plots. 

	# w labels 
	yy = w_vec.astype(int) # w_vec was saved as floats
	ny = num_w
	no_labels = num_w # how many labels to see on axis x
	step_y = int(ny / (no_labels - 1)) # step between consecutive labels
	y_positions = np.arange(0,ny,step_y) # pixel count at label position
	y_labels = yy[::step_y] # labels you want to see
	
	# tol_cg labels 
	xx = tol_cg_vec
	nx = num_tol_cg
	no_labels = num_tol_cg # how many labels to see on axis x
	step_x = int(nx / (no_labels - 1)) # step between consecutive labels
	x_positions = np.arange(0,nx,step_x) # pixel count at label position
	x_labels = xx[::step_x] # labels you want to see
	
	#------------------------------------------------------------
	# v 
	plt.figure(8)
	plt.imshow(v_norm, cmap=plt.cm.Blues,interpolation='nearest');
	
	#print np.format_float_scientific(np.float32(np.pi))
	#x_labels = np.format_float_scientific(x_labels, exp_digits=1)
	plt.yticks(y_positions, y_labels)
	plt.xticks(x_positions, x_labels, fontsize = smallerfont)

	plt.colorbar(format='%.0e') # pl.colorbar(myplot, format='%.0e')
	plt.xlabel('Rel. Tolerance CG') 
	plt.ylabel(r'Sketch Dim. $w$')
	plt.title(r'$|v|_2$')
	
	#plt.show()
	plt.tight_layout()
	plt.savefig(directory +'/ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_8_im_v.pdf', format='pdf')

	#------------------------------------------------------------
	# heatmaps (inner iterations, kappas)
	#------------------------------------------------------------
	# cg inner iterations (max over all outer)
	plt.figure(9)
	plt.imshow(it_cg_ipm, cmap=plt.cm.Blues,interpolation='nearest');
	plt.yticks(y_positions, y_labels) # (from above) 
	plt.xticks(x_positions, x_labels, fontsize = smallerfont) # (from above)
	#plt.colorbar(ticks=[15,20,25,30]) # for ARCENE
	plt.colorbar()
	plt.xlabel('Rel. Tolerance CG') 
	plt.ylabel(r'Sketch Dim. $w$')
	plt.title('Max. Inner Iterations')
	#plt.show()
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_9_cg_iter.pdf', format='pdf')

	# condition number (max over all outer)
	plt.figure(10)
	plt.imshow(kap_ipm, cmap=plt.cm.Blues,interpolation='nearest');
	plt.yticks(y_positions, y_labels) # (from above) 
	plt.xticks(x_positions, x_labels, fontsize = smallerfont) # (from above)
	plt.colorbar()
	plt.xlabel('Rel. Tolerance CG') 
	plt.ylabel(r'Sketch Dim. $w$')
	plt.title('Max. Cond. Num.')
	#plt.show()
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/w_tol_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_10_kap.pdf', format='pdf')

	# #------------------------------------------------------------
	# # it_ipm
	# plt.figure(9)
	# plt.imshow(it_ipm, cmap=plt.cm.Blues,interpolation='nearest');
	# plt.yticks(y_positions, y_labels) # (from above) 
	# plt.xticks(x_positions, x_labels, fontsize = smallerfont) # (from above)
	# plt.colorbar()
	# plt.xlabel('Rel. Tolerance CG') 
	# plt.ylabel(r'Sketch Dim. $w$')
	# plt.title('Outer Iterations')
	# #plt.show()
	# plt.tight_layout()
	# plt.savefig(directory +'ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_9_im_iter.pdf', format='pdf')



def plot_it_mul_w(m,n,data_desc, \
	iter_out_stan,iter_in_cg_vec_stan,iter_out_ipm_2,iter_in_cg_vec_ipm_2,iter_out_ipm_5, \
	iter_in_cg_vec_ipm_5,iter_out_ipm_8,iter_in_cg_vec_ipm_8, \
	kap_AD_vec,kap_ADW_vec_2,kap_ADW_vec_5,kap_ADW_vec_8,v_vec_2,v_vec_5,v_vec_8):
	#------------------------------------------------------------
	# Plots for Figure 1 (with three w's)
	#------------------------------------------------------------
	# Plot
	msize = 7
	colors = ['bo-','ro-','go-','co-','mo-','ko-','yo-']
	colors_blues = [(0, 0, 1),(0.4, 0.6, 0.9),(0.5, 0.5, 1),(0.11, 0.56, 1),(0.53, 0.80, 0.98)]  # dark to light blues


	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	## for Palatino and other serif fonts use:
	#rc('font',**{'family':'serif','serif':['Palatino']})
	#

	#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	
	# rc('font',**{'family':'serif','serif':['Times']})
	#rc('text', usetex=True)
	#hfont = {'fontname':'Helvetica'}
	plt.rcParams['font.size'] = 20
	smallerfont = 16 # override if needed smaller 
	#------------------------------------------------------------
	# 0 - Outer vs. Inner Iterations
	plt.figure(0)
	plt.semilogy(range(iter_out_stan), iter_in_cg_vec_stan, 'ro-', label='Stand. IPM', markersize=msize)

	plt.semilogy(range(iter_out_ipm_2), iter_in_cg_vec_ipm_2, 'o-', color=colors_blues[0], label='Sk. IPM w=200', markersize=msize)
	plt.semilogy(range(iter_out_ipm_5), iter_in_cg_vec_ipm_5, 'o-', color=colors_blues[1], label='Sk. IPM w=400', markersize=msize)
	plt.semilogy(range(iter_out_ipm_8), iter_in_cg_vec_ipm_8, 'o-', color=colors_blues[2], label='Sk. IPM w=1000', markersize=msize)
	
	plt.legend(loc='upper left', fontsize = smallerfont) # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel('Inner CG Iterations')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_mul_w_0.eps', format='eps', dpi=1200)
	
	#------------------------------------------------------------
	# 1 - kappa
	plt.figure(1)
	#plt.semilogy(range(iter_out_stan), kap_AD_vec, 'ro-', label=r'$ \kappa(AD^2A^T)$', markersize=8)
	plt.semilogy(range(iter_out_stan), kap_AD_vec, 'ro-', label='Stand. IPM', markersize=msize)

	#plt.semilogy(range(iter_out_ipm_2), kap_ADW_vec_2, 'o-', color=colors_blues[0], label=r'$\kappa(Q^{-1}AD^2A^T)$ w=200', markersize=msize)
	#plt.semilogy(range(iter_out_ipm_5), kap_ADW_vec_5, 'o-', color=colors_blues[1], label=r'$\kappa(Q^{-1}AD^2A^T)$ w=400', markersize=msize)
	#plt.semilogy(range(iter_out_ipm_8), kap_ADW_vec_8, 'o-', color=colors_blues[2], label=r'$\kappa(Q^{-1}AD^2A^T)$ w=1000', markersize=msize)
	
	plt.semilogy(range(iter_out_ipm_2), kap_ADW_vec_2, 'o-', color=colors_blues[0], label='Sk. IPM w=200', markersize=msize)
	plt.semilogy(range(iter_out_ipm_5), kap_ADW_vec_5, 'o-', color=colors_blues[1], label='Sk. IPM w=400', markersize=msize)
	plt.semilogy(range(iter_out_ipm_8), kap_ADW_vec_8, 'o-', color=colors_blues[2], label='Sk. IPM w=1000', markersize=msize)
	
	#plt.semilogy(range(iter_out_stan), kap_AD_vec, 'ro-', label=r'$ \kappa(AD^2A^T)$ Stand. IPM', markersize=8)

	#plt.semilogy(range(iter_out_ipm_2), kap_ADW_vec_2, 'o-', color=colors_blues[0], label=r'$\kappa(Q^{-1}AD^2A^T)$ Sketch IPM w = 200', markersize=8)
	#plt.semilogy(range(iter_out_ipm_5), kap_ADW_vec_5, 'o-', color=colors_blues[1], label=r'$\kappa(Q^{-1}AD^2A^T)$ Sketch IPM w = 500', markersize=8)
	#plt.semilogy(range(iter_out_ipm_8), kap_ADW_vec_8, 'o-', color=colors_blues[2], label=r'$\kappa(Q^{-1}AD^2A^T)$ Sketch IPM w = 1000', markersize=8)
	
	#ax.set_yscale('log')
	plt.legend(loc='upper left', fontsize = smallerfont) # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel('Condition Number')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_mul_w_1.eps', format='eps', dpi=1200)
	

	# #------------------------------------------------------------
	# # 1 - kappa formating
	# plt.figure(1)
	# plt.semilogy(range(iter_out_stan), kap_AD_vec, 'ro-', label='Stand. IPM', markersize=8)

	# plt.semilogy(range(iter_out_ipm_2), kap_ADW_vec_2, 'o-', color=colors_blues[0], label=r'Sk. IPM $w=200$', markersize=msize)
	# plt.semilogy(range(iter_out_ipm_5), kap_ADW_vec_5, 'o-', color=colors_blues[1], label=r'Sk. IPM $w=400$', markersize=msize)
	# plt.semilogy(range(iter_out_ipm_8), kap_ADW_vec_8, 'o-', color=colors_blues[2], label=r'Sk. IPM $w=1000$', markersize=msize)
	
	# #plt.semilogy(range(iter_out_stan), kap_AD_vec, 'ro-', label=r'$ \kappa(AD^2A^T)$ Stand. IPM', markersize=8)

	# #plt.semilogy(range(iter_out_ipm_2), kap_ADW_vec_2, 'o-', color=colors_blues[0], label=r'$\kappa(Q^{-1}AD^2A^T)$ Sketch IPM w = 200', markersize=8)
	# #plt.semilogy(range(iter_out_ipm_5), kap_ADW_vec_5, 'o-', color=colors_blues[1], label=r'$\kappa(Q^{-1}AD^2A^T)$ Sketch IPM w = 500', markersize=8)
	# #plt.semilogy(range(iter_out_ipm_8), kap_ADW_vec_8, 'o-', color=colors_blues[2], label=r'$\kappa(Q^{-1}AD^2A^T)$ Sketch IPM w = 1000', markersize=8)
	
	# #ax.set_yscale('log')
	# plt.legend(loc='upper left', fontsize = smallerfont) # , fontsize = 18
	# plt.xlabel('Outer Iterations')
	# plt.ylabel('Condition Number')  # ,format='%.e'
	# #plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	# plt.grid(True)
	# plt.tight_layout()
	# plt.savefig('/home/ubuntu/ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_mul_w_11.eps', format='eps', dpi=1200)
	

	#------------------------------------------------------------
	# 2 - v
	plt.figure(2)
	#plt.semilogy(range(iter_out_ipm), v_vec, colors[0], markersize=8)
	
	plt.semilogy(range(iter_out_ipm_2), v_vec_2, 'o-', color=colors_blues[0], markersize=msize)
	plt.semilogy(range(iter_out_ipm_5), v_vec_5, 'o-', color=colors_blues[1], markersize=msize)
	plt.semilogy(range(iter_out_ipm_8), v_vec_8, 'o-', color=colors_blues[2], markersize=msize)
	
	#ax.set_yscale('log')
	#plt.legend(loc='upper left') # , fontsize = 18
	plt.xlabel('Outer Iterations')
	plt.ylabel(r'$\|v\|_2$')  # ,format='%.e'
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(directory +'ipm_figures/svm_'+str(data_desc)+'_m'+str(m)+'_n'+str(n)+'_it_mul_w_2.eps', format='eps', dpi=1200)
	


