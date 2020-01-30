#!/usr/bin/env
#------------------------------------------------------------
import cvxpy as cvx
#import multiprocessing as mp
#from multiprocessing import Pool
#import string
import time
import numpy as np
import scipy as sp
#from scipy import optimize
from scipy import optimize as op
from scipy.io import loadmat
#from scipy.sparse import rand, find, spdiags, linalg, csr_matrix
import matplotlib
matplotlib.use('Agg') # needed for using matplotlib on remote machine
import matplotlib.pyplot as plt
#------------------------------------------------------------
import gen_data 	
import ipm_func
import svm_func
from parameters import m,n,p,w,gamma,sigma_step,sigma,t_iter, MAXIT_cg,tol_cg
#------------------------------------------------------------
np.random.seed(0) # reset random seed 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision=4)
#------------------------------------------------------------
# IPM main gen data example -- 1 Run
#------------------------------------------------------------


#------------------------------------------------------------
if __name__ == '__main__':
	print '---------------------------', '\n'
	print 'Test IPM', '\n'
	print '---------------------------', '\n'
	print '(m x n) = (', m, 'x', n, ')'
	print 'p = ', p
	print 'w = ', w
	print '\n', '---------------------------'
	#------------------------------------------------------------


	#------------------------------------------------------------
	# Data 
	t_data_1 = time.time()
	A, b, c = gen_data.gen_data_dense() 
	#print 'A = \n', A #, b #, c
	#print 'b = \n', b
	#print 'c = \n', c  
	t_data_2 = time.time()
	print 'time data generate    = ', t_data_2 - t_data_1, ' secs'

	# returns a dict
	#A_dict = loadmat('/home/ubuntu/ipm_data/demo/A.mat')
	#b_dict = loadmat('/home/ubuntu/ipm_data/demo/b.mat')
	#c_dict = loadmat('/home/ubuntu/ipm_data/demo/c.mat')
	#A = A_dict["A"]
	#b = b_dict["b"]
	#c = c_dict["c"]
	#b = np.squeeze(b)
	#c = np.squeeze(c)
	#m,n = np.shape(A)
	#w = 8
	

	#------------------------------------------------------------
	# CVX LP

	t_cvx_1 = time.time()
	x_cvx, p_cvx = svm_func.run_CVXPY_LP(A,b,c)
	t_cvx_2 = time.time()
	print'\ncvx = ', p_cvx
	print 'time cvx    = ', t_cvx_2 - t_cvx_1, ' secs'
	#------------------------------------------------------------
	# linprog

	t_linprog_1 = time.time()
	res = op.linprog(c, A_ub=A, b_ub=b,bounds=[0, None])
	t_linprog_2 = time.time()
	#print'\nlinprog = ', res
	# print 'time lpg    = ', t_linprog_2 - t_linprog_1, ' secs'
	#------------------------------------------------------------
	# IPM Standard
	t_stan_1 = time.time()
	x,y,s,t_iter_stan,t_ls_stan = ipm_func.ipm_standard(m,n,A,b,c)
	t_stan_2 = time.time()
	
	p_ipm_vec_stan = np.dot(c,x)		
	d_ipm_vec_stan = np.dot(b,y)	
	#------------------------------------------------------------
	# IPM Ours
	t_ipm_1 = time.time()
	x,y,s,t_iter_ipm,t_ls_ipm,iter_cg_ipm,v_vec = ipm_func.ipm(m,n,w,A,b,c)
	t_ipm_2 = time.time()
	
	p_ipm_vec = np.dot(c,x)		
	d_ipm_vec = np.dot(b,y)	
	#print 'x = \n',x
	#print 'x_lp = \n', res.x

	#------------------------------------------------------------
	print '\n-------------------------- \nResults: \n--------------------------'
	print 'time stan t_ls= ', t_ls_stan, ' secs'
	print 'time ipm t_ls = ', t_ls_ipm, ' secs'
	print 'speed up      = ', t_ls_stan/t_ls_ipm

	print '\niter stan = ', t_iter_stan
	print 'iter ipm  = ', t_iter_ipm

	#print '\niter cg ipm  = ', iter_cg_ipm
	#print 'iter cg stan = ', iter_cg_stan

	print '\n--------------------------'
	print 'time ipm      = ', t_ipm_2 - t_ipm_1, ' secs'
	print 'time ipm stan = ', t_stan_2 - t_stan_1, ' secs'
	print 'time cvx      = ', t_cvx_2 - t_cvx_1, ' secs'
	print 'time linprog  = ', t_linprog_2 - t_linprog_1, ' secs'
	print 'p   = ',p_ipm_vec
	print 'd   = ',d_ipm_vec
	print 'p   = ',p_ipm_vec_stan
	print 'd   = ',d_ipm_vec_stan
	print 'cvx = ',p_cvx

	#print 'lpg = ', res.fun
	# print 'linprog message= ', res.message
