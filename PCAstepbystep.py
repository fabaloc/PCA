# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:20:09 2019

@author: aciccone
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Retrieve data, in this case from an Excel file
path = '~/yourdata'
d = pd.read_excel(path, sheet_name='sheet')


# Store data in a pandas DataFrame and center data
d = pd.DataFrame(d)
mean_vec = np.mean(d, axis=0)  # mean of each variable 
var_vec = np.var(d,axis=0)
n,m = d.shape

d_std = d - mean_vec   #center data


# Covariance matrix
cov_mat = (d_std.T.dot(d_std))/ (n - 1) 


# Eigen-decomposition of the covariance matrix

eigval, eigvec = np.linalg.eig(cov_mat)


# check eigenvectors have unit length 1
for i in eigvec:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(i))
    print('Everything ok!')
    

# Sorting eigenvectors according to their eigenvalues

idx = eigval.argsort()[::-1]
eigval = eigval[idx]
eigvec = eigvec[:, idx]
tot = sum(eigval)
var_exp = [i/tot*100 for i in sorted(eigval, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

PC =  ['PC%s' %s for s in range(1, len(eigval) + 1)] #creating a vector of PCs names

# Plotting components and their explained variance
plt.scatter(var_exp, PC, alpha= 0.5)
plt.title('Explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.show()


# Selection of components

n_components = 3  #arbitrary number, it varies case by case
loadings = eigvec[:,:n_components:]


# Projection onto the new features space

scores = d_std.dot(loadings)


# Reconstructing original data and calculating the residuals

d_hat = scores.dot(loadings.T)
d_hat_raw = d_hat + mean_vec.values
res = d - d_hat_raw