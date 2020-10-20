# parameters.py

directory = '/Users/palma/Documents/Work/1_Projects/code_py/ipm/'
#--------------------------------------------------
# General Random data
# m << n, and  m < w < n  Examples:	(10,70,100), (100,300,10^6)

# SVM Random data
m = 10;      # number of training points
w = 70;
N = 100;      # dimension of x_i data point; dim of features
m_test = m;   # number of testing  points 
n = 2*N+1;    # dimension of final LP (m x n)

#--------------------------------------------------
# for ipm 
gamma = 0.5
sigma_step = 0.9; # scale alpha
sigma = 0.5; # smaller better, but require more iterations  

# Outer iterations 
#t_iter = 25 # use one of (t_iter or tol_ipm)
tol_ipm_mu = 1e-9 # tolerance for mu 
#tol_ipm_pd = 1e-3# abs difference between primal and dual residuals 

# CG parameters
power = 3 # (written like this so that we can do a str2num more easily)
tol_cg = pow(10,-1*power) # 1e-3
MAXIT_cg = 10000; 

#--------------------------------------------------
# for data matrix A generation
noise_c = 0.1; #  Ax <= b +/- noise_c
mu_data, sigma_data = 0, 0.1 # distribution from which a_ij are drawn
p = 0.8; # sparsity of matrix A  # 2/float(n); #0.05;

#--------------------------------------------------
# SVM data
DENSITY = 0.5 # sparsity of the "true signal" w_true

#--------------------------------------------------


