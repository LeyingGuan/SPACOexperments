import spaco as spaco
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib import patches as mpatches

file = "RealData/processed_IMPACTdata.npz"
data = np.load(file, allow_pickle = True)
X = data['X']
Z = data['Zstd']
Y = data['Y']
O = data['O']
Z0 = data['Z']
columns_immune = data['columns_immune'][:,0]
Ts = data['time_stamps']
T1 = np.sqrt(np.arange(len(Ts)))
spaco.seed_everything(seed=2023)
##rank selection
ranks = np.arange(1,11)
negliks = spaco.rank_selection_function(X = X, O =O, Z =Z,
                                        time_stamps = T1, ranks=ranks, early_stop = True,
                            max_iter = 30, cv_iter = 10, add_std = 0.0)

I = X.shape[0]
means = negliks.mean(axis = 0)
means=means[~np.isnan(means)]
idx_min = np.argmin(means)
rank_min = ranks[idx_min]
rank = rank_min

data_obj = dict(X=X, O=O, Z=Z,
                time_stamps=T1, rank=rank)

spaco_fit = spaco.SPACOcv(data_obj)
spaco_fit.train_preparation(run_prepare=True,
                        run_init=True,
                        mean_trend_removal=False,
                        smooth_penalty=True)

spaco_fit.train(update_cov=True,
            update_sigma_mu=True,
            update_sigma_noise=True,
            lam1_update=True, lam2_update=True,
            max_iter=30, min_iter=1,
            tol=1e-4, trace=True
            )

spaco_fit.beta

train_ids, test_ids = spaco.cutfoldid(n = I, nfolds = 5, random_state = 2022)

spaco_fit.cross_validation_train( train_ids,
                               test_ids,
                               max_iter=10,
                               min_iter = 1,
                               tol=1e-3,
                               trace = True)


data_obj['Z']=None
spacoNULL_fit = spaco.SPACOcv(data_obj)
spacoNULL_fit.train_preparation(run_prepare=True,
                        run_init=True,
                        mean_trend_removal=False,
                        smooth_penalty=True)

spacoNULL_fit.train(update_cov=True,
            update_sigma_mu=True,
            update_sigma_noise=True,
            lam1_update=True, lam2_update=True,
            max_iter=30, min_iter=1,
            tol=1e-4, trace=True
            )

###Compare SPACO and SPACOnull in data.frame
mu = spaco_fit.mu.copy()
columns_covariate = ['COVIDRISK_1',
                     'COVIDRISK_2',
                     'COVIDRISK_3', 'COVIDRISK_4',
                     'COVIDRISK_5',
                      'Age', 'sex', 'BMI']

columns_response = ['ICU', 'Clinicalscore', 'LengthofStay','Coagulopathy']

A = scipy.stats.spearmanr(mu, b=Y, axis=0, nan_policy='omit')
Acor = A.correlation
Apval = A.pvalue

result_dict = dict()
for j in np.arange(mu.shape[1]):
    result_dict['C'+str(j+1)+'cor'] = Acor[j,mu.shape[1]:]
    result_dict['C' + str(j + 1) + 'pval'] = Apval[j, mu.shape[1]:]

result_table_SPACO = pd.DataFrame.from_dict(result_dict)
result_table_SPACO.index = columns_response

mu0 = spacoNULL_fit.mu.copy()
tmp_align = scipy.stats.spearmanr(mu, b=mu0, axis=0, nan_policy='omit').correlation
tmp_align=tmp_align[rank:,:rank]
idx_align = [None]*rank
for j in np.arange(rank):
    idx_align[j] = np.argmax(tmp_align[:,j])

mu0 = mu0[:,idx_align]
A = scipy.stats.spearmanr(mu0, b=Y, axis=0, nan_policy='omit')
Acor = A.correlation
Apval = A.pvalue

result_dict_null = dict()
for j in np.arange(mu0.shape[1]):
    result_dict_null['C'+str(j+1)+'cor'] = Acor[j,mu0.shape[1]:]
    result_dict_null['C' + str(j + 1) + 'pval'] = Apval[j, mu0.shape[1]:]

result_table_SPACO_null = pd.DataFrame.from_dict(result_dict_null)

result_table_SPACO_null.index = columns_response

result_table_response = pd.concat([result_table_SPACO.copy(), result_table_SPACO_null.copy()], axis=0, keys=["SPACO", "SPACO-"])
result_table_response=result_table_response.round({'C1cor': 2, 'C2cor': 2,'C3cor': 2, 'C4cor': 2})

for j in np.arange(4):
    result_table_response['C'+str(j+1)+'pval'] = result_table_response['C'+str(j+1)+'pval'].apply(
        lambda val: '%.1E' % val)

result_table_response
print(result_table_response.to_latex(escape=False))

################Testing for Z
delta = 0;
tol = 0.01;
fixbeta0 = False;
method = "cross"
nfolds = 5
random_state = 0
spaco.seed_everything(seed=2023)
feature_eval = spaco.CRtest_cross(spaco_fit, type=method, delta=delta)
#precalculate quantities that stay the same for different Z
feature_eval.precalculation()
feature_eval.cut_folds(nfolds=nfolds, random_state=random_state)
feature_eval.beta_fun_full(nfolds=nfolds, max_iter=1, tol= tol, fixbeta0=fixbeta0)
#drop each feature and refit
for j in np.arange(feature_eval.Z.shape[1]):
    feature_eval.beta_fun_one(nfolds=nfolds, j=j, max_iter=1, fixbeta0=fixbeta0)
#calculate the test statistics for conditional and mariginal independence
for j in np.arange(feature_eval.Z.shape[1]):
    feature_eval.precalculation_response(j=j)
    feature_eval.coef_partial_fun(j = j, inplace=True)
    feature_eval.coef_marginal_fun(j = j, inplace=True)

dataZ_random = np.load("RealData/randmoized_covariates.npz")
Zmarginal_random = dataZ_random['Zmarginal_random']
Zpartial_random = dataZ_random['Zconditional_random']
Zpartial_random=Zpartial_random
Zmarginal_random=Zmarginal_random

feature_eval.coef_partial_random = np.zeros((feature_eval.coef_partial.shape[0],
                                             feature_eval.coef_partial.shape[1],
                                             Zpartial_random.shape[2]))
feature_eval.coef_marginal_random = np.zeros((feature_eval.coef_partial.shape[0],
                                             feature_eval.coef_partial.shape[1],
                                             Zmarginal_random.shape[2]))


for j in np.arange(feature_eval.Z.shape[1]):
    feature_eval.coef_random_fun(Zpartial_random, j, type = "partial")
    feature_eval.coef_random_fun(Zmarginal_random, j, type="marginal")

pvals_partial_empirical, pvals_partial_fitted = \
    feature_eval.pvalue_calculation(type = "partial",pval_fit = False, dist_name ='nct')

pvals_marginal_empirical, pvals_marginal_fitted = \
    feature_eval.pvalue_calculation(type = "marginal",pval_fit = False, dist_name ='nct')

families = ['binomial', 'binomial', 'binomial', 'binomial', 'binomial',
            'gaussian', 'binomial', 'gaussian']

nz_count = np.zeros(Z.shape[1]) * np.nan
for j in np.arange(Z.shape[1]):
    if families[j] == "binomial":
        nz_count[j] = np.sum(Z0[:,j]==1)

def pvalue_correction_func(pvals):
    m = len(pvals)
    pval_ranks = scipy.stats.rankdata(pvals)
    pval1 = pvals * m / pval_ranks
    pval1[pval1>1]=1.0
    return pval1

pvalue_restable = pd.DataFrame({#'beta':spaco_fit.beta[:,2],
                                'pval(partial)':pvals_partial_empirical[:,2],
                                'pval(marginal)':pvals_marginal_empirical[:,2],
                                'padj(partial)':pvalue_correction_func(pvals_partial_empirical[:,2]),
                                'padj(marginal)':pvalue_correction_func(pvals_marginal_empirical[:,2]),
                                'nz_count':nz_count,
                                }, index = columns_covariate)
pvalue_restable
pvalue_restable=pvalue_restable.round(3)
print(pvalue_restable.to_latex(escape=False))
####Data overview#####
from matplotlib.gridspec import GridSpec
####data overview, 'IFNy', 'IL-18'
Zmat = pd.DataFrame(Z)
Ymat = pd.DataFrame(Y)
Ymat.columns = columns_response
Zmat.columns = columns_covariate
fig = plt.figure(figsize=(15, 7.5))
gs = GridSpec(nrows=2, ncols=4, left=0.05, right=.99,
              bottom=0.15, top=0.99)
fontsize = 15
labelsize = 15
names = ['TcellsofLivecells', 'TotalNeutrophilsofLivecells',
         'HLA.DR.ofTotalMono', 'IL6']
j0 = np.where(columns_immune == names[0])[0]
j1 = np.where(columns_immune == names[1])[0]
j2 = np.where(columns_immune == names[2])[0]
j3 = np.where(columns_immune == names[3])[0]
col_idx = Ymat['ICU']

axis1 = {}
axis1[0, 0] = fig.add_subplot(gs[0, 0])
axis1[0, 1] = fig.add_subplot(gs[0, 1])
axis1[1, 0] = fig.add_subplot(gs[1, 0])
axis1[1, 1] = fig.add_subplot(gs[1, 1])
for i in np.arange(X.shape[0]):
    x0 = X[i, :, j0].reshape(-1)
    x1 = X[i, :, j1].reshape(-1)
    x2 = X[i, :, j2].reshape(-1)
    x3 = X[i, :, j3].reshape(-1)
    o0 = O[i, :, 0]
    o0 = np.where(o0 == 1)[0]
    if len(o0) > 1:
        if col_idx[i] == 1.0:
            axis1[0, 0].plot(Ts[o0], x0[o0], color='b')
            axis1[0, 1].plot(Ts[o0], x1[o0], color='b')
            axis1[1, 0].plot(Ts[o0], x2[o0], color='b')
            axis1[1, 1].plot(Ts[o0], x3[o0], color='b')
        else:
            axis1[0, 0].plot(Ts[o0], x0[o0], color='r')
            axis1[0, 1].plot(Ts[o0], x1[o0], color='r')
            axis1[1, 0].plot(Ts[o0], x2[o0], color='r')
            axis1[1, 1].plot(Ts[o0], x3[o0], color='r')
    else:
        if col_idx[i] == 1.0:
            axis1[0, 0].scatter(Ts[o0], x0[o0], color='b',
                                s=1)
            axis1[0, 1].scatter(Ts[o0], x1[o0], color='b',
                                s=1)
            axis1[1, 0].scatter(Ts[o0], x2[o0], color='b',
                                s=1)
            axis1[1, 1].scatter(Ts[o0], x3[o0], color='b',
                                s=1)
        else:
            axis1[0, 0].scatter(Ts[o0], x0[o0], color='r',
                                s=1)
            axis1[0, 1].scatter(Ts[o0], x1[o0], color='r',
                                s=1)
            axis1[1, 0].scatter(Ts[o0], x2[o0], color='r',
                                s=1)
            axis1[1, 1].scatter(Ts[o0], x3[o0], color='r',
                                s=1)

axis1[1, 0].set_xlabel('DFSO', fontsize=fontsize)
axis1[1, 1].set_xlabel('DFSO', fontsize=fontsize)
axis1[0, 0].set_ylabel('observed', rotation=90,
                       fontsize=fontsize)
axis1[1, 0].set_ylabel('observed', rotation=90,
                       fontsize=fontsize)
axis1[0, 0].text(20, 2.5, names[0], fontsize=fontsize)
axis1[0, 1].text(2, 2.5, names[1], fontsize=fontsize)
axis1[1, 0].text(10, 2.5, names[2], fontsize=fontsize)
axis1[1, 1].text(40, 2.5, names[3], fontsize=fontsize)

###percent of in-sample variance explained
# reshape Xhat to a matrix
X3 = spaco_fit.intermediantes['Xmod2']
O3 = spaco_fit.intermediantes['O2']
X3hat = np.zeros(X3.shape)
T = len(Ts)
UPhi = np.zeros((I * len(Ts), spaco_fit.K))
for k in np.arange(spaco_fit.K):
    UPhi[:, k] = np.kron(spaco_fit.Phi[:, k].reshape((T, 1)),
                         spaco_fit.mu[:, k].reshape(
                             (I, 1))).reshape(-1)

for j in np.arange(X3.shape[0]):
    X3hat[j, :] = np.matmul(UPhi, spaco_fit.V[j, :])

X3hat = np.transpose(X3hat[:, O3[0, :] == 1])
X3 = np.transpose(X3[:, O3[0, :] == 1])
X3hat = pd.DataFrame(X3hat, columns=columns_immune)
X3 = pd.DataFrame(X3, columns=columns_immune)

axis2 = {}
axis2[0, 0] = fig.add_subplot(gs[0, 2])
axis2[0, 1] = fig.add_subplot(gs[0, 3])
axis2[1, 0] = fig.add_subplot(gs[1, 2])
axis2[1, 1] = fig.add_subplot(gs[1, 3])
##observed ~ predicted
Xarray_hat = np.zeros(X.shape)
for j in np.arange(X.shape[2]):
    Xarray_hat[:, :, j] = np.transpose(
        np.matmul(UPhi, spaco_fit.V[j, :]).reshape((T, I)))

####data overview, 'IFNy', 'IL-18'
names = ['TcellsofLivecells', 'TotalNeutrophilsofLivecells',
         'HLA.DR.ofTotalMono', 'IL6']
j0 = np.where(columns_immune == names[0])[0]
j1 = np.where(columns_immune == names[1])[0]
j2 = np.where(columns_immune == names[2])[0]
j3 = np.where(columns_immune == names[3])[0]
col_idx = Ymat['ICU']-1
fig1, axis1 = plt.subplots(2, 2)
for i in np.arange(X.shape[0]):
    x0 = X[i, :, j0].reshape(-1)
    x1 = X[i, :, j1].reshape(-1)
    x2 = X[i, :, j2].reshape(-1)
    x3 = X[i, :, j3].reshape(-1)
    z0 = Xarray_hat[i, :, j0].reshape(-1)
    z1 = Xarray_hat[i, :, j1].reshape(-1)
    z2 = Xarray_hat[i, :, j2].reshape(-1)
    z3 = Xarray_hat[i, :, j3].reshape(-1)
    o0 = O[i, :, 0]
    o0 = np.where(o0 == 1)[0]
    x0 = x0[o0];
    x1 = x1[o0];
    x2 = x2[o0];
    x3 = x3[o0];
    z0 = z0[o0];
    z1 = z1[o0];
    z2 = z2[o0];
    z3 = z3[o0];
    if len(o0) > 1:
        if col_idx[i] <= 0:
            ll0 = np.argsort(np.argsort(z0))
            axis2[0, 0].plot(z0[ll0], x0[ll0], color='b')
            ll0 = np.argsort(np.argsort(z1))
            axis2[0, 1].plot(z1[ll0], x1[ll0], color='b')
            ll0 = np.argsort(np.argsort(z2))
            axis2[1, 0].plot(z2[ll0], x2[ll0], color='b')
            ll0 = np.argsort(np.argsort(z3))
            axis2[1, 1].plot(z3[ll0], x3[ll0], color='b')
        else:
            ll0 = np.argsort(np.argsort(z0))
            axis2[0, 0].plot(z0[ll0], x0[ll0], color='r')
            ll0 = np.argsort(np.argsort(z1))
            axis2[0, 1].plot(z1[ll0], x1[ll0], color='r')
            ll0 = np.argsort(np.argsort(z2))
            axis2[1, 0].plot(z2[ll0], x2[ll0], color='r')
            ll0 = np.argsort(np.argsort(z3))
            axis2[1, 1].plot(z3[ll0], x3[ll0], color='r')
    else:
        if col_idx[i] <= 0:
            axis2[0, 0].scatter(z0, x0, color='b', s=1)
            axis2[0, 1].scatter(z1, x1, color='b', s=1)
            axis2[1, 0].scatter(z2, x2, color='b', s=1)
            axis2[1, 1].scatter(z3, x3, color='b', s=1)
        else:
            axis2[0, 0].scatter(z0, x0, color='r', s=1)
            axis2[0, 1].scatter(z1, x1, color='r', s=1)
            axis2[1, 0].scatter(z2, x2, color='r', s=1)
            axis2[1, 1].scatter(z3, x3, color='r', s=1)

axis2[0, 0].text(-1.5, 2.5, names[0], fontsize=fontsize)
axis2[0, 1].text(-1.5, 2.5, names[1], fontsize=fontsize)
axis2[1, 0].text(-2, 2.5, names[2], fontsize=fontsize)
axis2[1, 1].text(-1.7, 2.5, names[3], fontsize=fontsize)

axis2[1, 0].set_xlabel('estimated', fontsize=fontsize)
axis2[1, 1].set_xlabel('estimated', fontsize=fontsize)

axis2[1, 0].tick_params(axis='both', labelsize=labelsize)
axis2[1, 1].tick_params(axis='both', labelsize=labelsize)
axis2[0, 0].tick_params(axis='both', labelsize=labelsize)
axis2[0, 1].tick_params(axis='both', labelsize=labelsize)

axis1[1, 0].tick_params(axis='both', labelsize=labelsize)
axis1[1, 1].tick_params(axis='both', labelsize=labelsize)
axis1[0, 0].tick_params(axis='both', labelsize=labelsize)
axis1[0, 1].tick_params(axis='both', labelsize=labelsize)

# axis2[0,0].set_ylabel('observed', rotation = 90)
# axis2[0,1].set_ylabel('observed', rotation = 90)
fig.savefig('RealData/IMPACT_overview.png')
plt.close()


##feature investigation
# feature investigation
idx_selected = [1,2]
J = spaco_fit.X.shape[2]
K = spaco_fit.K
cor_mat_features = np.zeros((J,K))
pval_mat_features = np.zeros((J,K))
for j in np.arange(J):
    for k in np.arange(K):
        z = np.kron(spaco_fit.Phi[:,k].reshape((T,1)), spaco_fit.mu[:,k].reshape(I,1)).reshape(-1)
        x = spaco_fit.intermediantes['Xmod2'][j,:]
        o = spaco_fit.intermediantes['O2'][j,:]
        tmp = scipy.stats.pearsonr(z[o==1], x[o==1])
        cor_mat_features[j,k] = tmp[0]
        pval_mat_features[j,k] = tmp[1]



ntop = 25
top_features = np.zeros((ntop, 3,K), dtype = object)
Vuse = -spaco_fit.V.copy()
Vuse0 = -np.abs(Vuse)


#Vuse0 =pval_mat_features
for k in np.arange(K):
    #plt.hist(-np.log10(pval_mat_features[k]))
    z0 = Vuse0[:,k]
    z = Vuse[:,k]
    #z = spaco.V[:,k]
    thr = np.sort(z0)[ntop]
    idx = np.where((z0 < thr))[0]
    feature_order = np.argsort(z0[idx])
    top_features[:,0,k] = columns_immune[idx][feature_order]
    top_features[:,1,k] = z[idx][feature_order]
    top_features[:,2,k] = -cor_mat_features[idx,k][feature_order]

###coefficient contribution plot
x = 24-np.arange(25)
##for clinical
k = 1
coef_factor = top_features[:,1,k]
negatives = np.zeros(coef_factor.shape)
negatives[coef_factor<0] = coef_factor[coef_factor<0]
positives = np.zeros(coef_factor.shape)
positives[coef_factor>0] = coef_factor[coef_factor>0]

figsize = (40, 20)
fig = plt.figure(figsize=figsize)

margins = {  #     vvv margin in inches
    "left"   :     .1 / figsize[0],
    "bottom" :     .1 / figsize[1],
    "right"  : 1 - .1 / figsize[0],
    "top"    : 1-.1  / figsize[1]
}
plt.subplot(211)
plt.barh(x, positives,  height=1, color='r')
plt.barh(x, negatives,  height=1, color='b')
plt.ylim([-.5, 24.5])
plt.xticks(fontsize = 30)
plt.yticks(x, top_features[:,0,k], fontsize = 20)
plt.title('Component'+str(k+1), fontsize = 50)

k = 2
coef_factor = top_features[:,1,k]
negatives = np.zeros(coef_factor.shape)
negatives[coef_factor<0] = coef_factor[coef_factor<0]
positives = np.zeros(coef_factor.shape)
positives[coef_factor>0] = coef_factor[coef_factor>0]
plt.subplot(212)
plt.barh(x, positives,  height=1, color='r')
plt.barh(x, negatives,  height=1, color='b')

plt.ylim([-.5, 24.5])
plt.xticks(fontsize = 30)
plt.yticks(x, top_features[:,0,k], fontsize = 20)
plt.title('Component '+str(k+1), fontsize = 50)

plt.savefig('RealData/feature_contribution_IMPACT.png')

plt.close()
