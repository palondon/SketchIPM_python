#!/usr/bin/env
#------------------------------------------------------------
import time
import numpy as np
import scipy as sp
from parameters import p,gamma,sigma_step,sigma,tol_ipm_mu,MAXIT_cg

#------------------------------------------------------------
# functions:
# ipm (our Sketched IPM, using PCG)
# ipm_standard (standard IPM, using CG internally)
#------------------------------------------------------------
def ipm(m,n,w,A,b,c,tol_cg):
	#------------------------------------------------------------
	# IPM (our Sketched IPM, using PCG)

	print '\n------------------------------------------' 
	print 'Sketched IPM' 
	print '------------------------------------------' 

	# initial point 
	#x = np.ones(n)
	#s = np.ones(n)
	#y = np.ones(m)

	zeta = 10000
	x = zeta*np.ones(n)
	s = zeta*np.ones(n)
	y = np.zeros(m)


	# check feasible; initial point 
	feas_p = abs(np.dot(A,x) - b)
	feas_d = abs(np.dot(np.transpose(A),y) - c)

	if np.ndarray.min(feas_p) > 1E-3:
		print 'initial point: feasible p'
	else: 
		print 'initial point: infeasible p'

	if np.ndarray.min(feas_d) > 1E-3:
		print 'initial point: feasible d'
	else: 
		print 'initial point: infeasible d'


	p_ipm_vec = []  # np.zeros(size_vec)		
	d_ipm_vec = [] 	
	time_ls_vec = [] 
	iter_in_cg_vec = [] 
	kap_ADW_vec = [] 
	v_vec = []

	#p_ipm_vec = np.zeros(size_vec)		
	#d_ipm_vec = np.zeros(size_vec)	
	#t_ls_vec = np.zeros(size_vec)	
	#iter_cg_ipm_vec = np.zeros(size_vec)
	#v_vec = np.zeros(size_vec)	
	
	#-----------------------------------------------------
    # (c) generate W 
	W = np.random.normal(0, 1, (n,w))/np.sqrt(w)


	mu = 1
	k = 0 	
	while (mu > tol_ipm_mu): 	#for k in range(t_iter):
		print '\n------------------------------------------'
		print 'k = ', k
		print '\nw = ', w
		print 'tol_cg = ', tol_cg
		t_tot_1 = time.time()
		#-------------------------------------------------
		# (a) calculate residuals wrt the current point (x,y,s)
		r_p = np.dot(A,x) - b
		r_d = np.dot(np.transpose(A),y) + s - c

		#-------------------------------------------------
		# (b) update matrices 
		d = np.sqrt(x) * np.sqrt(1/s) # * is element-wise multiply
		s_inv = 1/s
		mu = np.dot(x,s)/n

		p_ipm = -r_p - sigma*mu*np.dot(A,s_inv) + np.dot(A,x) - np.dot(A,d*d*r_d)

		B = np.dot(A,np.diag(d))

		#-----------------------------------------------------
	    # (c) generate W 
		#W = np.random.normal(0, 1, (n,w))/np.sqrt(w)

		#-------------------------------------------------
		# (d) solve the system for del_y_hat

		#-------------------------------------------------
		# invert Q_inv  (need for f_tilde comp)
		WW = np.dot(W,np.transpose(W))
		Q = np.dot(np.dot(B,WW),np.transpose(B)) #Q = B*(W*W')*B'
		U_Q,S_Q,V_Q = np.linalg.svd(Q) # returns the transpose of V #Q_inv = V_Q*diag(1./diag(S_Q))*U_Q'

		Q_inv = np.dot(np.dot(V_Q, np.diag(1/S_Q)), np.transpose(U_Q))  # V_Q transpose here?

		#-------------------------------------------------
		# QR of (B*W)', used to make preconditioner: M = L*L' , where L = R'
		Q_qr,R = sp.linalg.qr(np.transpose(np.dot(B,W)), mode='economic')
		L = np.transpose(R) # for clariry L = R^T
		#-------------------------------------------------
		# input to cg:   scipy.sparse.linalg.LinearOperator
		BB = np.dot(B,np.transpose(B))

		def mv(v):
			return np.dot(BB,v)

		def mv_L(v):
			return np.dot(L, np.dot(np.transpose(L),v))

		BB_linoper = sp.sparse.linalg.LinearOperator((m,m), matvec=mv)
		M_linoper = sp.sparse.linalg.LinearOperator((m,m), matvec=mv_L)

		lhs = np.dot(Q_inv, BB)
		rhs = np.dot(Q_inv, p_ipm) 

		pre_con = np.linalg.inv(np.dot(L,np.transpose(L)))
		#inv_L = np.linalg.inv(L)
		#pre_con = np.dot(inv_L,np.transpose(inv_L))
		#-------------------------------------------------
		# CG    
		#-------------------------------------------------
		# Matlab: [del_y_hat_cg,FLAG,RELRES,iter_cg] = pcg(@(x) B * (B' * x), p_ipm, tol_cg, MAXIT_cg, L, L');
		
		# condition number 
		e = sp.linalg.eigh(BB, Q, eigvals_only = 1)
		condi = np.ndarray.max(e)/np.ndarray.min(e)
		kap_ADW_vec.append(condi)
		
		print 'kap_AD = ', condi

		t_cg_1 = time.time()
		# call cg
		#num_iters_cg = 0

		#del_y_hat_cg, num_iters_cg = sp.sparse.linalg.cg(BB, p_ipm, x0=None, tol=tol_cg, maxiter=MAXIT_cg, M=np.dot(L,np.transpose(L)))# callback=None, atol=None)
		
		del_y_hat_cg,status,num_iters_cg = pcg_solve(BB, p_ipm, tol_cg, MAXIT_cg, pre_con)

		#del_y_hat_cg,status,num_iters_cg = pcg_solve(BB_linoper, p_ipm, tol_cg, MAXIT_cg, Q_inv)

		#del_y_hat_cg,status,num_iters_cg = pcg_solve(BB_linoper, p_ipm, tol_cg, MAXIT_cg, Q_inv)
		#del_y_hat_cg,status,num_iters_cg = pcg_solve(BB_linoper, p_ipm, tol_cg, MAXIT_cg, L)
		#del_y_hat_cg,status,num_iters_cg = pcg_solve(BB_linoper, p_ipm, tol_cg, MAXIT_cg, M_linoper)
		#del_y_hat_cg, iter_cg_ipm_vec[k] = sp.sparse.linalg.cg(BB_linoper, p_ipm, x0=None, tol=tol_cg, maxiter=MAXIT_cg, M=np.dot(L, np.transpose(L)))# callback=None, atol=None)
		#del_y_hat_cg, status = sp.sparse.linalg.cg(BB_linoper, p_ipm, x0=None, tol=tol_cg, maxiter=MAXIT_cg, M=M_linoper) # callback=callback
		t_cg_2 = time.time()
		#print 'time cg    = ', t_cg_2 - t_cg_1, ' secs'
		print '\nCG iter    = ', num_iters_cg

		iter_in_cg_vec.append(num_iters_cg)

		#-------------------------------------------------
		# linsolve 
		#-------------------------------------------------
		t_ls_1 = time.time()
		#del_y_hat_linsolve = np.linalg.solve(lhs, rhs) # del_y_hat_linsolve = linsolve(lhs, rhs);

		del_y_hat_linsolve = np.linalg.solve(BB, p_ipm) # del_y_hat_linsolve = linsolve(lhs, rhs);
		t_ls_2 = time.time()
		time_ls_vec.append(t_ls_2 - t_ls_1)
		#print 'time ls    = ', t_ls_2 - t_ls_1, ' secs'

		#-------------------------------------------------
		del_y_hat = del_y_hat_cg
		#del_y_hat = del_y_hat_linsolve
		#-------------------------------------------------
		# (f) compute v 

		t_svd_1v = time.time()

		XS_inv2 = np.diag(1/np.sqrt(x*s)) 
		f_tilde = np.dot(lhs, del_y_hat) - rhs # f_tilde = lhs*del_y_hat_cg - rhs 
		CW = np.dot(np.dot(B,XS_inv2),W) # B(XS)^-1/2 W

		# calculate pinv of (B*XS_inv2*W)
		U_CW,S_CW,V_CW = np.linalg.svd(CW)
		S_CW_padded = np.concatenate((np.diag(1/S_CW), np.zeros((w-m,m))),axis = 0)
		#print np.shape(CW) # CW is (m x w)

		CW_pinv = np.dot(np.dot(V_CW, S_CW_padded), np.transpose(U_CW))
		v = np.dot(np.dot(W, CW_pinv), f_tilde) #v = W*pinv(B*XS_inv2*W)*f_tilde;

		v_vec.append(np.linalg.norm(v,2))
		print "\n|v|_2 = ", np.linalg.norm(v,2)
		# if were to use pinv()
		#v = np.dot(np.dot(W, np.linalg.pinv(CW)), f_tilde) #v = W*pinv(B*XS_inv2*W)*f_tilde;

		t_svd_2v = time.time()
		#print 'time cal v = ', t_svd_2v - t_svd_1v, ' secs'

		# check rank
		if(abs(np.linalg.matrix_rank(B) - m) > 1e-2): 
			print "\nRank(B) is NOT == m. \n"

		if(abs(np.linalg.matrix_rank(CW) - m) > 1e-2): 
			print "\nRank(CW) is NOT == m. \n"
		#-------------------------------------------------
		# (f) compute del_s and del_x
		# del_s_hat = - r_d - (A'*del_y_hat);
		# del_x_hat = - x + sigma*mu*s_inv.*ones(n,1) - d.*d.*del_s_hat - s_inv.*v;

		del_s_hat = -r_d - np.dot(np.transpose(A),del_y_hat)

		del_x_hat = -x + sigma*mu*s_inv - d*d*del_s_hat - s_inv*v
		#del_x_hat = -x + sigma*mu*s_inv - d*d*del_s_hat

		#------------------------------------------------------
		# step size
		alpha_p = 1
		alpha_d = 1

		for j in range(n): 
			if (x[j] + alpha_p*del_x_hat[j] < 0):
				alpha_p = -x[j]/del_x_hat[j]
			if (s[j] + alpha_d*del_s_hat[j] < 0):
				alpha_d = -s[j]/del_s_hat[j]

		alpha_p = min(alpha_p*sigma_step, 1)
		alpha_d = min(alpha_d*sigma_step, 1)

		alpha = min(alpha_p, alpha_d)


		x = x + alpha*del_x_hat
		s = s + alpha*del_s_hat
		y = y + alpha*del_y_hat

		bool_x = any(x < 0)
		bool_s = any(s < 0)

		if (bool_x & bool_s): # check if any x_i < 0 and s_i < 0
			print bool_x & bool_s

		t_tot_2 = time.time()
		#------------------------------------------------------
		# Record
		p_hat = np.dot(c,x)	
		d_hat = np.dot(b,y)	
		p_ipm_vec.append(p_hat)	
		d_ipm_vec.append(d_hat)	

		# update 
		error_pd = abs(p_hat - d_hat)

		print '\np = ',p_hat
		print   'd = ',d_hat
		print   'error_pd = ',error_pd
		print   'mu = ',mu

		# update
		k = k + 1
	    #-------------------------------------------------
	iter_out = k

	# feasibility report at end
	if np.ndarray.min(feas_p) > 1E-3:
		print 'feasible p'
	else: 
		print 'infeasible p'

	if np.ndarray.min(feas_d) > 1E-3:
		print 'feasible d'
	else: 
		print 'infeasible d last step, but mu < mu_tol achieved'


	# convert result lists to arrays
 	iter_in_cg_vec = np.asarray(iter_in_cg_vec)
 	kap_ADW_vec = np.asarray(kap_ADW_vec, dtype=np.float32)
 	v_vec = np.asarray(v_vec, dtype=np.float32)
 	time_ls_vec = np.asarray(time_ls_vec, dtype=np.float32)

 	time_ls = time_ls_vec.sum() # total time
	return x,y,s,iter_out,iter_in_cg_vec,kap_ADW_vec,v_vec,time_ls



#-------------------------------------------------
num_iters = 0 # global variable 
#-------------------------------------------------
def pcg_solve(A, b, tol_cg, MAXIT_cg, M_linOp):
	# pre-conditioner: is M=np.dot(L, np.transpose(L))
	# function needed so that we can retrieve number of iterations used by CG 
	# (default info returned by CG is 0/-1/etc int indicating success / failure)
	# this modifies the "callback" functionality of CG
	global num_iters # needed to indicate that want to change the global variable 
	num_iters = 0
	def callback(xk):
		global num_iters
		#print 'num_iters = ' , num_iters
		num_iters += 1

	# call the solver 
	x, status = sp.sparse.linalg.cg(A, b, x0=None, tol=tol_cg, maxiter=MAXIT_cg, M=M_linOp, callback=callback)
	#x, status = sp.sparse.linalg.cg(A, b, x0=None, tol=tol_cg, maxiter=MAXIT_cg, M=np.dot(L, np.transpose(L)), callback=callback)

	return x, status, num_iters

#-------------------------------------------------
def cg_solve(A, b, tol_cg, MAXIT_cg):
	# function needed so that we can retrieve number of iterations used by CG 
	# (default info returned by CG is 0/-1/etc int indicating success / failure)
	# this modifies the "callback" functionality of CG
	global num_iters # needed to indicate that want to change the global variable 
	num_iters = 0
	def callback(xk):
		global num_iters
		#print 'num_iters = ' , num_iters
		num_iters += 1

	# call the solver 
	x,status = sp.sparse.linalg.cg(A, b, x0=None, tol=tol_cg, maxiter=MAXIT_cg, callback=callback)
	return x, status, num_iters

#------------------------------------------------------------
def ipm_standard(m,n,A,b,c,tol_cg):
	#------------------------------------------------------------
	# Standard IPM (analogous to ours, without sketching)
	# solver used: CG
	#------------------------------------------------------------
	# IPM
	#x = np.ones(n)
	#s = np.ones(n)
	#y = np.ones(m)

	zeta = 10000
	x = zeta*np.ones(n)
	s = zeta*np.ones(n)
	y = np.zeros(m)

	#x = np.random.rand(n)
	#s = np.random.rand(n)
	#y = np.random.rand(m)

	# check feasible 
	feas_p = abs(np.dot(A,x) - b)
	feas_d = abs(np.dot(np.transpose(A),y) - c)

	if np.ndarray.min(feas_p) > 1E-3:
		print 'feasible p'
	else: 
		print 'infeasible p'

	if np.ndarray.min(feas_d) > 1E-3:
		print 'feasible d'
	else: 
		print 'infeasible d'

	p_ipm_vec = []  	
	d_ipm_vec = [] 	
	time_ls_vec = [] 
	iter_in_cg_vec = [] 
	kap_AD_vec = [] 
	# note: use lists rather than ndarrays; doesn't require contiguous block of RAM 
	# so append doesn't make a new contiguous block each time. 
	# convert to ndarrays at the end 
	print '\n------------------------------------------' 
	print 'Standard IPM' 
	print '------------------------------------------' 
	mu = 1
	k = 0 
	while (mu > tol_ipm_mu): 
		print '\n------------------------------------------' 
		print 'k = ', k
		print 'tol_cg = ', tol_cg
		#-------------------------------------------------
		# (a) calculate residuals wrt the current point (x,y,s)
		r_p = np.dot(A,x) - b
		r_d = np.dot(np.transpose(A),y) + s - c

		#-------------------------------------------------
		# (b) update matrices 
		d2 = x * (1/s) # * is element-wise multiply
		s_inv = 1/s
		mu = np.dot(x,s)/n

		lhs = np.dot(np.dot(A,np.diag(d2)),np.transpose(A))
		rhs = -r_p + np.dot(A, x - sigma*mu*(1/s) - np.dot( np.diag(x * s_inv),r_d)  )
		
		#-------------------------------------------------
		# condition number of ADDA
		#kap_AD_vec.append(np.linalg.cond(np.dot(A,np.diag(np.sqrt(x) * (1/np.sqrt(s))))))

		# condition number 
		#e = sp.linalg.eigvals(lhs)
		e = sp.linalg.eigh(lhs, eigvals_only = 1)
		condi = np.ndarray.max(e)/np.ndarray.min(e)
		kap_AD_vec.append(condi)

		print 'kap_AD = ', kap_AD_vec[k]
		#-------------------------------------------------
		# CG (will compare with linsolve) 
		def mv(v):
			return np.dot(lhs,v)
		#def matvec_transp(v): # mv_t
		#	return
		def rmv(v):
			return np.dot(v, np.transpose(lhs))
		AA = sp.sparse.linalg.LinearOperator((m,m), matvec=mv, rmatvec =rmv,dtype='float64')

		x0 = np.empty(m)
		#print np.shape(AA)
		#print np.shape(rhs)
		#print np.shape(x0)

		#temp = sp.sparse.csr_matrix(AA)
		#info, iter_pcg, relres = krylov.pcg(AA, rhs, x0, tol_cg, MAXIT_cg)

		iter_in_cg_k = 0
		t_cg_1 = time.time()
		del_y_hat_cg_stan, status, iter_in_cg_k = cg_solve(AA, rhs, tol_cg, MAXIT_cg)
		t_cg_2 = time.time()
		print '\niter_cg_stan = ', iter_in_cg_k
		print 'time cg stan = ', t_cg_2 - t_cg_1, ' secs'

		iter_in_cg_vec.append(iter_in_cg_k)
		#-------------------------------------------------
		# linsolve
		#t_ls_1 = time.time()
		#del_y_hat_linsolve = np.linalg.solve(lhs, rhs)
		#t_ls_2 = time.time()
		#time_ls_vec.append(t_ls_2 - t_ls_1)
		#print 'time ls     = ', t_ls_2 - t_ls_1, ' secs'

		#-------------------------------------------------
		del_y_hat = del_y_hat_cg_stan
		#del_y_hat = del_y_hat_linsolve
		#-------------------------------------------------
		# (f) compute del_s and del_x
		del_s_hat = -r_d - np.dot(np.transpose(A),del_y_hat)
		# del_x = -X*Sinv*del_s + tau*(1./s) - x;
		#del_x_hat = - np.dot(np.dot(np.diag(x), np.diag(s_inv)),del_s_hat) + sigma*mu*s_inv - x
		del_x_hat = - (x * s_inv * del_s_hat) + sigma*mu*s_inv - x

		#------------------------------------------------------
		# step size
		alpha_p = 1
		alpha_d = 1

		for j in range(n): 
			if (x[j] + alpha_p*del_x_hat[j] < 0):
				alpha_p = -x[j]/del_x_hat[j]
			if (s[j] + alpha_d*del_s_hat[j] < 0):
				alpha_d = -s[j]/del_s_hat[j]

		alpha_p = min(alpha_p*sigma_step, 1)
		alpha_d = min(alpha_d*sigma_step, 1)

		alpha = min(alpha_p, alpha_d)

		x = x + alpha*del_x_hat
		s = s + alpha*del_s_hat
		y = y + alpha*del_y_hat
		#print x
		#print s
		#print y

		bool_x = any(x < 0)
		bool_s = any(s < 0)

		if (bool_x & bool_s): # check if any x_i < 0 and s_i < 0
			print bool_x & bool_s

		#------------------------------------------------------
		# Record
		p_hat = np.dot(c,x)
		d_hat = np.dot(b,y)
		p_ipm_vec.append(p_hat)	
		d_ipm_vec.append(d_hat)	

		# update 
		error_pd = abs(p_hat - d_hat)

		print '\np = ',p_hat
		print 'd = ',d_hat
		print 'error_pd = ',error_pd
		print 'mu = ',mu

		# update
		k = k + 1
	    #-------------------------------------------------
 	iter_out = k

 	iter_in_cg_vec = np.asarray(iter_in_cg_vec)
 	kap_AD_vec = np.asarray(kap_AD_vec, dtype=np.float32)
 	time_ls_vec = np.asarray(time_ls_vec, dtype=np.float32)

	time_ls = time_ls_vec.sum() # total time
	return x,y,s,iter_out,iter_in_cg_vec,kap_AD_vec,time_ls



