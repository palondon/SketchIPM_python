README
---------------------------------------------------------------
1/29/2020 - modified 10/16/2020 
Palma London

Code for "Sketched IPM algorithm" proposed in this work:

Faster Randomized Infeasible Interior Point Methods for Tall/Wide Linear Programs
A. Chowdhuri, P. London, H. Avron, and P. Drineas,
34th Conference on Neural Information Processing Systems (NeurIPS), accepted, 2020.

---------------------------------------------------------------
About the code:
---------------------------------------------------------------

1. To run a small example on synthetic SVM data, run: 

main_1_data_syn.py 

2. Parameter settings are found in: 

parameters.py 

---------------------------------------------------------------
Algorithm implementation: 
---------------------------------------------------------------
Implementation of our "Sketched IPM with PCG" and a "Standard IPM using CG" are both found in:

ipm_func.py 


---------------------------------------------------------------
Synthetic data generation: 
---------------------------------------------------------------
- gen_data.py (generate random LP instances)

- parameters.py (The user can set various problem dimensions (m, n) and sketching dimension 'w', to experiment on various synthetic data matrices)


---------------------------------------------------------------
Real data examples
---------------------------------------------------------------

load_svm_real_data.py (load and reformat real data matrices for SVM problems)

Due to the large size of the datasets, we don't include that datasets in this directory. The user of this code must download the datasets themselves from the UCI Machine Learning Repository.  For example, the'ARCENE' dataset is found here: 
https://archive.ics.uci.edu/ml/datasets/Arcene

---------------------------------------------------------------
To generate specific Figures in the paper, run:
---------------------------------------------------------------

-- main_1_data_mul_w.py 

Figure 1 (a) Outer Iteration vs Inner Iteration (CG or PCG iterations) and 
Figure 1 (b) Outer Iteration vs Condition number 

(Figure 1 highlights the ARCENE dataset, but 'main_1data_mul_w.py' code can be run on any dataset. For more, please see comments in the code. We experimented on the data sets, and the results are reported in the paper, in Table 1:
'ARCENE'
'DEXTER'
'DrivFace'
'DOROTHEA' 

-- main_w_tolcg_grid.py 

Figure 1 (c) Heat map of CG tolerance, Sketching dimension, and Max. Inner Iterations 
Figure 1 (d) Heat map of CG tolerance, Sketching dimension, and Max. Condition Number

As for main_1data_mul_w.py, this code can also be applied to any of the data sets mentioned above. For more, please see comments in the code.


---------------------------------------------------------------
Other subroutines: 
---------------------------------------------------------------

- svm_func.py (functions for SVM related tasks)

- sim_func.py (subroutines needed to carry out the simulation)

- cvxpy_LP.py (code to solve a given LP using cvxpy (to use as a reference other than python's internal LP solver))

- the other files (named main_test*.py) can be ignored for now; they are small tests

---------------------------------------------------------------
Plot and Load saved output 
---------------------------------------------------------------

- main_load_plot.py

- plot_func.py (subroutines for generating the plots)



---------------------------------------------------------------


In this subsection we explore the limit to which we can push not solving each linear system to high accuracy. We allow PCG to have a relatively low tolerance. In this case, the norm of the vector v will also grow. 





