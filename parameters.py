# parameters.py

#--------------------------------------------------
# General Random data
# m << n, and  m < w < n 	(5,45,50) (10,100,60)
#m = 10;   #    5  10  
#w = 0;   #   45  60  sketching dim     
#n = 100;  #   50 100    

# SVM Random data
m = 100;      # number of training points
w = 700;
N = 1000;      # dimension of x_i data point; dim of features
m_test = m;   # number of testing  points 
n = 2*N+1;    # dimension of final LP (m x n)

#--------------------------------------------------
# for ipm 
gamma = 0.5
sigma_step = 0.9; # scale alpha
sigma = 0.5; # smaller better, but require more iterations  

# Outer iterations 
t_iter = 25 # use one of (t_iter or tol_ipm)
tol_ipm = 1e-3 # abs difference between primal and dual residuals 

# CG parameters
tol_cg = 1e-4; 
MAXIT_cg = 1000000; # extreme but see that python's CG sometimes even maxs this out...

#--------------------------------------------------
# for data matrix A generation
noise_c = 0.1; #  Ax <= b +/- noise_c
mu_data, sigma_data = 0, 0.1 # distribution from which a_ij are drawn
p = 0.8; # sparsity of matrix A  # 2/float(n); #0.05;

#--------------------------------------------------
# SVM data
DENSITY = 0.5 # sparsity of the "true signal" w_true

#--------------------------------------------------
#num_processes = 32


