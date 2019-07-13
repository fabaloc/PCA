
'''                                             PRINCIPAL COMPONENT ANALYSIS                                                  '''


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Import Spread data for all currently traded indexes

import os
cwd = os.getcwd()
 
df = pd.read_excel(r'~/Itraxxdata_22Mar19_04Jun19.xlsx',  sheet_name='Itraxx')


# Explore dataset

df.head()
df.tail()
print(df.dtypes)
df.isnull().values.sum()   # Number of NaN in dataset


# modify dates format and create vector of dates

df['Date'] = pd.to_datetime(df.Date)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
date = pd.to_datetime(df.iloc[:,0])
date = date.dt.strftime('%Y-%m-%d')

# renaming columns and indexing with dates

df = df.iloc[:, 1::].set_index(date)

col_names = ['S31_3Y', 'S31_5Y', 'S31_7Y', 'S31_10Y','S30_3Y', 'S30_5Y', 'S30_7Y', 'S30_10Y',
             'S28_3Y', 'S28_5Y', 'S28_7Y', 'S28_10Y', 'S26_3Y',
             'S26_5Y', 'S26_7Y', 'S26_10Y', 'S24_5Y', 'S24_7Y',
             'S24_10Y']

df.columns = col_names


neworder = ['S26_3Y', 'S24_5Y', 'S28_3Y', 'S30_3Y','S26_5Y',
     'S31_3Y', 'S24_7Y', 'S28_5Y', 'S26_7Y',
     'S30_5Y', 'S31_5Y',  'S28_7Y', 'S24_10Y',  
     'S30_7Y', 'S31_7Y', 'S26_10Y', 'S28_10Y',
     'S30_10Y', 'S31_10Y']  # ordering by ascending expiry

df = df.reindex(columns=neworder)


# let's take a look at the Itraxx curve for each date

plt.figure(figsize=(8,4))
plt.plot(df.T)
plt.xticks(rotation=45)   

# Creating of a dataframe containing just daily changes in spread

diff_df = df.iloc[::-1, :].diff()  # difference data starting from the farthest date
diff_df = diff_df.set_index(date[::-1])  # indexing with dates
diff_df = diff_df[1::].sort_index(ascending=False, axis=0)  # sorting dataset back from most recent to farthest date
diff_df = diff_df.loc[~(diff_df == 0).any(axis=1)] # Exclude vacation days

newdate = list(diff_df.index)
newdate.append(date[len(date)-1])
newdate = pd.to_datetime(newdate)

multiplier =np.zeros(len(newdate)-1)
for i in range(len(newdate)-1):
    delta = (newdate[i] - newdate[i+1]).days
    multiplier[i] = 1/(np.sqrt(((delta)/365) * 260))
    
d = diff_df.mul(multiplier, axis=0)

summary = d.describe()


'''                                                             Rolling PCA                                                   '''

n, m = d.shape
t = 20
dates = date[0:n-t:]

scores1 = np.zeros((n-t))  # scores
scores2 = np.zeros((n-t))
scores3 = np.zeros((n-t))

coeff1 = pd.DataFrame(data = np.zeros((n-t, m))) # loadings
coeff2 = pd.DataFrame(data = np.zeros((n-t, m)))
coeff3 = pd.DataFrame(data = np.zeros((n-t, m)))
    
eigval1 = np.zeros((n-t))
eigval2 = np.zeros((n-t))
eigval3 = np.zeros((n-t))

explained_variancePC1 = np.zeros((n-t))
explained_variancePC2 = np.zeros((n-t))
explained_variancePC3 = np.zeros((n-t))


R = pd.DataFrame(data = np.zeros((n-t, m)))  # Residuals

scaledcoeff1 = pd.DataFrame(data = np.zeros((n-t, m))) # loadings
scaledcoeff2 = pd.DataFrame(data = np.zeros((n-t, m)))
scaledcoeff3 = pd.DataFrame(data = np.zeros((n-t, m)))


for i in range(n-t):
    
    # calculation of mean, std_dev, scores and loadings for each window
    d_roll = d.iloc[i:i+t,:]
    dstd_roll = StandardScaler(with_std=False).fit_transform(d.iloc[i:i+t,:])
    mean = np.mean(d.iloc[i:i+t,:], axis=0)
    stdev = np.sqrt(np.var(d.iloc[i:i+t, :], axis=0))
    pca = PCA(n_components=3)
    scores = pca.fit_transform(dstd_roll)
    loadings = pca.components_
    eigvalues = pca.explained_variance_
    expvarratio = pca.explained_variance_ratio_ * 100
    Xstd = pca.inverse_transform(scores)  # std data reconstruction using PCs
    Xraw = Xstd + mean.values  # original data reconstruction
    rollres = d_roll - Xraw
   
    # store PCs of the most recent day in each window
    scores1[i] = scores[0, 0]
    scores2[i] = scores[0, 1]
    scores3[i] = scores[0, 2]
    
    # storing eigenvectors for each index for each window
    coeff1.iloc[i, :] = loadings[0, :]
    coeff2.iloc[i, :] = loadings[1, :]
    coeff3.iloc[i, :] = loadings[2, :]
    
    # store eigen-values of the most recent day of each window
    eigval1[i] = eigvalues[0]
    eigval2[i] = eigvalues[1]
    eigval3[i] = eigvalues[2]
    
    # variance explained in each day
    explained_variancePC1[i] = expvarratio[0]
    explained_variancePC2[i] = expvarratio[1]
    explained_variancePC3[i] = expvarratio[2]
    
    # Store residuals of the most recent day for each index in each window
    R.iloc[i, :] = rollres.iloc[0, :].values
    

# Scaling Coefficients to get an idea of shifts in the curve by one stdev

scaledcoeff1 = coeff1 * np.std(scores1)
scaledcoeff2 = coeff2 * np.std(scores2)
scaledcoeff3 = coeff3 * np.std(scores3)

# Scaling scores in the same fashion

scaled_score1 = scores1 / np.std(scores1)
scaled_score2 = scores2 / np.std(scores2)
scaled_score3 = scores3 / np.std(scores3)
scaledscores = pd.DataFrame({'scaledscore1': scaled_score1,
                             'scaledscore2': scaled_score2,
                             'scaledscore3': scaled_score3})
scaledscores = scaledscores.set_index(dates)

# Creating dataframes with dates and columns names
scores = pd.DataFrame({'scores1': scores1, 'scores2': scores2, 'scores3': scores3}, index=dates)
coeff1 = coeff1.set_index(dates)
coeff1.columns = d.columns
coeff2 = coeff2.set_index(dates)
coeff2.columns = d.columns
coeff3 = coeff3.set_index(dates)
coeff3.columns =d.columns
scaledcoeff1 = scaledcoeff1.set_index(dates)
scaledcoeff1.columns = d.columns
scaledcoeff2 = scaledcoeff2.set_index(dates)
scaledcoeff2.columns = d.columns
scaledcoeff3 = scaledcoeff3.set_index(dates)
scaledcoeff3.columns = d.columns
R = R.set_index(dates)
R.columns = d.columns


#Export results to Excel

from openpyxl import load_workbook
 
PCA_ItraxxMain = df.to_excel(r'~\PCA_ItraxxMain.xlsx',
                                    index = True , header = True, sheet_name='Original_Data')

path = r'~\PCA_ItraxxMain.xlsx'
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine='openpyxl')
writer.book = book
d.to_excel(writer, sheet_name='DailyChange')
scaledscores.to_excel(writer, sheet_name='Weights_Betas')
scaledcoeff1.to_excel(writer, sheet_name='ScaledCoeff1')
scaledcoeff2.to_excel(writer, sheet_name='ScaledCoeff2')
scaledcoeff3.to_excel(writer, sheet_name='ScaledCoeff3')
R.to_excel(writer, sheet_name='Residuals')
writer.save()


# Visualization

#1 Shape of coeffiecients for the most recent day and all tenors

plt.plot(coeff1.iloc[0,:], label='PC1')  
plt.plot(coeff2.iloc[0,:], label='PC2')
plt.plot(coeff3.iloc[0,:], label='PC3')
plt.legend()


#2 Shape of coefficents for S31, the most liquid index

coeff1_s31 = coeff1[['S31_3Y', 'S31_5Y', 'S31_7Y', 'S31_10Y']]
coeff1_s31 = coeff1_s31[::-1]
coeff2_s31 = coeff2[['S31_3Y', 'S31_5Y', 'S31_7Y', 'S31_10Y']]
coeff2_s31 = coeff2_s31[::-1]
coeff3_s31 = coeff3[['S31_3Y', 'S31_5Y', 'S31_7Y', 'S31_10Y']]
coeff3_s31 = coeff3_s31[::-1]


plt.plot(coeff1_s31.iloc[0,:], label='PC1')  
plt.plot(coeff2_s31.iloc[0,:], label='PC2')
plt.plot(coeff3_s31.iloc[0,:], label='PC3')
plt.legend()


#2 Residuals for S31, the most liquid index

res_s31 = R[['S31_3Y', 'S31_5Y', 'S31_7Y', 'S31_10Y']]
res_s31 = res_s31[::-1]

plt.plot(res_s31.iloc[:, 0], label='res31_3Y')  
plt.plot(res_s31.iloc[:, 1], label='res31_5Y')
plt.plot(res_s31.iloc[:, 2], label='res31_7Y')
plt.plot(res_s31.iloc[:, 3], label='res31_10Y')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.6))
plt.xticks(np.arange(0, len(R), step=3))
plt.xlabel('Date')
plt.ylabel('Residual')
plt.gcf().autofmt_xdate()
plt.show()


