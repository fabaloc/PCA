# PCA
Experimenting with Principal Components Analysis:

# PCA step by step 
The py file 'PCAstepbystep' is for (personal) pedagogical goals and shows all the steps that need to be followed to do a Principal Component Analysis. The method used is the eigendecomposition of the covariance matrix. Alternatively Singular Value Decomposition could be used to reach similar results.

# Rolling PCA on CDS data
The py file 'RollingPCAonCDS' shows an application of PCA on CDS data using a rolling window of 20 trading days. The first three Principal components are used to estimate parallel shifts, change in slope and curvature of the European Investment Grade CDS Index (Itraxx - Main). The residuals, results of the difference between the original data and the reconstructed data, could be used as trading signals for a variety of strategies: a test of trading strategies based on Residuals represents material for further expansion of the analysis.
The data used can be found in the file 'Itraxxdata_22Mar19_04Jun19.xlsx' that covers daily moves of the whole curve from March 22nd, 2019 to June 4th, 2019.
