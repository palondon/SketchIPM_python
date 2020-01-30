#------------------------------------------------------------
import random
import Queue
import numpy as np
import scipy as sp
from numpy import linalg as LA
from scipy.sparse import rand, find, spdiags, linalg, csr_matrix
#------------------------------------------------------------
from parameters import m,n,p,gamma,noise_c, N, m_test
#------------------------------------------------------------
# gen_data
# various functions for generating random instances (A,b,c) of LPs 
#------------------------------------------------------------

def gen_data_dense(): # dense, uniform 
	# A: Gaussian
	# x: Uniform
	# b = A*x + uniform noise
	# c: Gaussian

	#A = np.random.random((m,n)) 	   # Uniform 
	A = np.random.normal(0, 1, (m,n))  # Gaussian 
	B = np.random.choice([0, 1], size=(m,n), p=[1-p,p]) # make p-sparse 
	A = np.multiply(A,B)

	#np.fill_diagonal(A, 1) # if want to change the diagonal; ensure A is not low rank

	A[m-1,:] = np.ones(n) # ensured boundedness 

	x_true = np.random.rand(n) # Uniform
	#x_true = np.random.normal(0, 1, (n,1)) # Gaussian

	b = np.dot(A, x_true) + noise_c*np.random.rand(m) # Uniform 
	#c = np.random.rand(n)+ noise_c # Uniform 
	c = np.random.normal(0, 1, n) # Gaussian 
	#c = np.ones((n,1))

	return A, b, c
#------------------------------------------------------------

def gen_data_SVM(m,n,p,DENSITY):
	# pass m,n b/c may want to vary them in a loop, override parameter.py file 
	TEST = m
	DENSITY = 0.99
	lambda_ours = 0.1
	lambd = lambda_ours
	# noise for data when making w_true
	offset = 0 # no bias term for now 
	sigma = 45
	#-------------------------------
	# true classifier w_true
	#-------------------------------
	w_true = np.random.randn(N) # note: don't want (N,1); creates extra dummy dimension 

	# set some of w_true to 0
	idxs = np.random.choice(range(N), int((1-DENSITY)*N), replace=False)
	for idx in idxs:
	    w_true[idx] = 0

	#-------------------------------
	# Data matrix (for Training)
	#-------------------------------
	X_train = np.random.normal(0, 1, size=(m,N)) #X = rand(m, n, density=p, format="csr") #
	# sign y_train's
	y_train = np.sign(X_train.dot(w_true) + offset + np.random.normal(0,sigma,size=m))

	#-------------------------------
	# More data (for Testing)
	#-------------------------------
	X_test = np.random.normal(0, 5, size=(m_test,N))
	y_test = np.sign(X_test.dot(w_true) + offset + np.random.normal(0,sigma,size=m_test))
	return X_train, y_train, X_test, y_test, w_true



def gen_data_1(): # sparse, plus noise 
	# 1. Diagonal, in a sparse matrix
	rand_diag = np.random.random((2,n))  # 3
	diags = np.array([0,1]) # -1
	A1 = spdiags(rand_diag, diags, m, n, format="csr") # format="csr"

	# 2. Sparse Noise 
	A2 = rand(m, n, density=p, format="csr") # Compressed Sparse Row matrix
	A = A1 + A2;

	x_true = np.random.random((n,1))
	b = sp.sparse.csr_matrix.dot(A, x_true) + noise_c*np.random.random((m,1))

	return A, b, x_true
#------------------------------------------------------------



def gen_data_match(): # matching, overdetermined
	# 1. Diagonal, put it in a sparse matrix
	rand_diag = np.random.random((2,n))  # 3
	diags = np.array([0,1]) # -1
	A1 = spdiags(rand_diag, diags, n, n, format="csr") # format="csr"
	# 2. Matching
	rand_diag2 = np.random.random((2,n))  # 3
	diags2 = np.array([0,n/2]) 
	A2 = spdiags(rand_diag2, diags2, n/2, n, format="csr") 
	A = sp.sparse.vstack([A1, A2]).tocsr()



	x_true = np.random.random((n,1))
	b = sp.sparse.csr_matrix.dot(A, x_true) + noise_c*np.random.random((m,1))
	#b = np.dot(A, x_true.tosprase()) # dont use np.dot, use  .. .csr_matrix.dot

	return A, b, x_true

#-------------------------------------------------
def load_data_nh2010():
	n = 48838 # variables and rows; symmetric 
	m = n
	cpt = 234550 # number of nonzeros in A
	matrix=[]
	with open('nh2010.txt') as f:
		for i in range(0):
			f.next()
		for line in f:
			matrix.append(line)

	print 'nh2010 data set loaded ( 48838 x 48838 )\n'
	#print 'len(matrix) = ', len(matrix)

	row_ind=[0]*len(matrix)
	col_ind=[0]*len(matrix)
	data=[0]*len(matrix)
	for i in range(len(matrix)):
	    full_array = np.fromstring(matrix[i], dtype=float, sep=" ")
	    row_ind[i]=full_array[0]
	    col_ind[i]=full_array[1]-1 # make up for 1 to 0 indexing 
	    data[i]=full_array[2]

	A = sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

	x_true = np.random.random((n,1))
	b = sp.sparse.csr_matrix.dot(A, x_true) + noise_c*np.random.random((m,1))

	#print 'A = \n', A[1:10] #.todense()
	#print 'b = \n', b[1:10] #.todense()

	return A, b, x_true, m, n


	




#-------------------------------------------------
# For my reference: 
#A = np.random.random((m,n))  # Uniform  # summer = sum(A) /m # check that average is 0.5
#A = np.random.normal(mu, sigma, (m,n)) # Gaussian 
#A = rand(m, n, density=0.25, format="csr", random_state=42) # Compressed Sparse Row matrix
#print 'A1 = \n', A1
#print 'A1 = \n', A1.todense()
#print 'A2 = \n', A2.todense()
# print 'A = \n', A.todense()
## print find(A)[0] # 1st result of find: array of indices of non zero row
## print find(A)[1] # 2nd result of find: array of indices of non zero cols
## print find(A)[2] # 3rd result of find: the nonzero entries
#nnz = len(find(A)[0]) # number of nonzeros 
#b = np.dot(A, x_true.tosprase()) # dont use np.dot, use  .. .csr_matrix.dot

##print 'nnz = ',nnz
##print type(nnz)
##print '(m*n) = ',(m*n)
#print 'percentage nnz = ', float(nnz)/float(m*n)


