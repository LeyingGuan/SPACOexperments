import pandas as pd
import numpy as np
import scipy

X = pd.read_csv("RealData/DatasetB.csv")
Xcomplete = pd.read_csv("RealData/DatasetBimputed.csv")

Xuse = X
tmp = Xuse['sex']
tmp[tmp == 'F'] = 0
tmp[tmp == 'M'] = 1
tmp = np.array(tmp, dtype=float)
Xuse['sex'] = tmp
idx_healthy = ['HCW' in x for x in Xuse['ID']]
idx_patient = [not x for x in idx_healthy]
X_patient_check = Xuse.loc[idx_patient]
X_healthy_check = Xuse.loc[idx_healthy]
columns_meta_check = Xuse.columns[:24]
columns_immune_check = Xuse.columns[24:]
Xmeta_healthy_check = X_healthy_check[columns_meta_check]
Xmeta_patient_check = X_patient_check[columns_meta_check]
Ximmune_healthy_check = X_healthy_check[
    columns_immune_check]
Ximmune_patient_check = X_patient_check[
    columns_immune_check]
idx_feature = np.where(np.mean(np.isnan(Ximmune_patient_check),
                       axis=0) <= 0.2)[0]

mm = np.mean(np.isnan(Ximmune_patient_check),axis=0)
plt.hist(mm, bins=30)

# Add labels and title
plt.xlabel('missing rate')
plt.ylabel('Count')
plt.title('Histogram')
plt.savefig('IMPACTmissing.png')
# Show the plot


Xuse = Xcomplete
##data preparation
tmp = Xuse['sex']
tmp[tmp == 'F'] = 0
tmp[tmp == 'M'] = 1
tmp = np.array(tmp, dtype=float)
Xuse['sex'] = tmp
idx_healthy = ['HCW' in x for x in Xuse['ID']]
idx_patient = [not x for x in idx_healthy]
X_patient = Xuse.loc[idx_patient]
X_healthy = Xuse.loc[idx_healthy]
columns_meta = Xuse.columns[:24]
columns_immune = Xuse.columns[24:]
Xmeta_healthy = X_healthy[columns_meta]
Xmeta_patient = X_patient[columns_meta]
Ximmune_healthy = X_healthy[columns_immune]
Ximmune_patient = X_patient[columns_immune]

Ximmune_patient = Ximmune_patient[columns_immune[idx_feature]]
columns_immune = columns_immune[idx_feature]
xmin = np.nanmin(Ximmune_patient_check)
xmax = np.nanmax(Ximmune_patient_check)

# create the 3D array and get the covariates
l = Ximmune_patient.shape[0]
Ximmune_patient0 = Ximmune_patient.copy()
normalizations = scipy.stats.norm.ppf(list((np.arange(l) + 0.5) / l))
for j in np.arange(Ximmune_patient.shape[1]):
    x = Ximmune_patient[columns_immune[j]].to_numpy()
    Ximmune_patient[columns_immune[j]] = normalizations[np.argsort(np.argsort(x))]

patient_ids = np.array([x[0] for x in Xmeta_patient['ID'].str.split('[.]')])
unique_patient_ids = np.unique(patient_ids)
ts = Xmeta_patient['DFSO']
ts = ts.to_numpy()
T0 = np.sort(np.unique(ts))
I = len(unique_patient_ids)
T = len(T0)
J = Ximmune_patient.shape[1]

columns_covariate = ['COVIDRISK_1',
                     'COVIDRISK_2',
                     'COVIDRISK_3', 'COVIDRISK_4',
                     'COVIDRISK_5',
                      'Age', 'sex', 'BMI']

columns_response = ['ICU', 'Clinicalscore', 'LengthofStay','Coagulopathy']

Z = Xmeta_patient[columns_covariate]
Z = Z.to_numpy()
Y = Xmeta_patient[columns_response]
Y = Y.to_numpy()
q = Z.shape[1]
Xmat = Ximmune_patient.to_numpy()

Xarray = np.zeros((I, T, J))
Xarray[:] = np.nan
Zarray = np.zeros((I, T, q))
Zarray[:] = np.nan
Yarray = np.zeros((I, T, len(columns_response)))
Yarray[:] = np.nan

for i in np.arange(I):
    lli = np.where(patient_ids == unique_patient_ids[i])[0]
    xi = Xmat[lli, :]
    zi = Z[lli, :]
    ts0 = ts[lli]
    if len(ts0) == 1:
        llt = np.where(T0 == ts0)[0]
        Xarray[i, llt, :] = Xmat[lli, :]
        Zarray[i, llt, :] = Z[lli, :]
        Yarray[i, llt, :] = Y[lli, :]
    else:
        for t in np.arange(len(ts0)):
            llt = np.where(T0 == ts0[t])[0]
            Xarray[i, llt, :] = Xmat[lli[t], :]
            Zarray[i, llt, :] = Z[lli[t], :]
            Yarray[i,llt,:] = Y[lli[t], :]

Obs = np.ones(Xarray.shape)
Obs[np.isnan(Xarray)] = 0
print(np.var(Ximmune_patient, axis=0))

Zmat = np.zeros((I, q))
Ymat = np.zeros((I, len(columns_response)))
Zmat[:] = np.nan
Ymat[:] = np.nan
for i in np.arange(I):
    for j in np.arange(q):
        ll = np.where(~ np.isnan(Zarray[i, :, j]))[0]
        if len(ll) == 0:
            Zmat[i, j] = 0
        else:
            Zmat[i, j] = Zarray[i, ll[0], j]
    for j in np.arange(Ymat.shape[1]):
        ll = np.where(~ np.isnan(Yarray[i, :, j]))[0]
        Ymat[i, j] = np.nanmax(Yarray[i, :, j])

Zmat[np.isnan(Zmat)] = 0

##standardize Z
Zstd = Zmat.copy()
for j in np.arange(q):
    Zstd[:, j] = (Zstd[:, j] - np.mean(
        Zstd[:, j])) / np.std(Zstd[:, j])

file = "RealData/processed_IMPACTdata.npz"
np.savez(file, Zstd = Zstd, Z = Zmat, Y = Ymat, X = Xarray, O = Obs, time_stamps = T0,
         columns_immune = np.array(columns_immune).reshape((len(columns_immune),1)))

##generate randomized Z and save for testing purpose
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import random
pandas2ri.activate()
stats = importr('stats')
base = importr('base')
glmnet_package = importr('glmnet')
glmnet = glmnet_package.glmnet
cv_glmnet = glmnet_package.cv_glmnet

random_num=2022
families = ['binomial', 'binomial', 'binomial', 'binomial', 'binomial',
            'gaussian', 'binomial', 'gaussian']
Zmat0 = Zmat

np.random.seed(random_num)
B = 2000; nfolds0 = 5
Zconditional = np.zeros((Zmat0.shape[0], Zmat0.shape[1], B))
for j in np.arange(Zmat0.shape[1]):
    y = Zmat0[:, j]
    x = np.delete(Zmat0, j, axis=1)
    fitted = cv_glmnet(x=x, y=y.reshape((len(y), 1)),
                       intercept=True,
                       nfolds=nfolds0, nlambda=100,
                       family=families[j])
    yhat = stats.predict(fitted, x, s='lambda.min',
                         type='response')
    if families[j] == 'binomial':
        for i in np.arange(Zmat0.shape[0]):
            Zconditional[i, j, :] = np.random.binomial(n=1,
                                                       p=
                                                       yhat[
                                                           i],
                                                       size=B)
    else:
        v = np.sqrt(np.mean((y - yhat) ** 2))
        for i in np.arange(Zmat0.shape[0]):
            Zconditional[i, j, :] = yhat[
                                        i] + np.random.normal(
                size=B) * v

##standardize
for b in np.arange(B):
    for j in np.arange(Zmat0.shape[1]):
        Zconditional[:, j, b] = (Zconditional[:, j,b] - np.mean(Zconditional[:, j, b])) / np.std(Zconditional[:, j, b])


np.random.seed(random_num)
Zmarginal = np.zeros((Zmat0.shape[0], Zmat0.shape[1], B))
list1 = list(np.arange(Zmat0.shape[0]))
for b in np.arange(B):
    list2 = random.sample(list1, Zmat0.shape[0])
    Zmarginal[:, :, b] = Zmat0[list2, :].copy()

##standardize
for b in np.arange(B):
    for j in np.arange(Zmat0.shape[1]):
        Zmarginal[:, j, b] = (Zmarginal[:, j, b] - np.mean(Zmarginal[:, j, b])) / np.std(Zmarginal[:, j, b])

idx = list(np.where(np.sum(np.isnan(Zconditional), axis = 0).sum(axis = 0) == 0)[0])
Zconditional =Zconditional[:,:,idx]

idx = list(np.where(np.sum(np.isnan(Zmarginal), axis = 0).sum(axis = 0) == 0)[0])
Zmarginal =Zmarginal[:,:,idx]

np.savez(file = "RealData/randmoized_covariates.npz" , Zconditional_random = Zconditional, Zmarginal_random = Zmarginal)
