import spaco
import pickle
import numpy as np
import copy
import importlib
import sys
importlib.reload(spaco)

#extract fitted object from reconstruction_comparison.py output
#I = 100; T = 30; q = 100;J = 10;SNR1 = 1.0;SNR2 = 3.0;rate = 0.1
I = int(sys.argv[1])
T = int(sys.argv[2])
J = int(sys.argv[3])
q = int(sys.argv[4])
rate = float(sys.argv[5])
SNR1 = float(sys.argv[6])
SNR2 = float(sys.argv[7])
iteration = 20
curpath = "/home/lg689/project/SPACOresults"

res_file_name = curpath + '/results/construction/data_subj' + str(
    I) + '_time' + str(T) + '_feature' + str(J) + \
                '_Z' + str(q) + '_rate' + str(
    rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
                str(SNR2) + '.pickle'

pvalue_file_name = curpath + '/results/test/data_subj' + str(
    I) + '_time' + str(T) + '_feature' + str(J) + \
                '_Z' + str(q) + '_rate' + str(
    rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
                str(SNR2) + '.pickle'

with open(res_file_name, 'rb') as handle:
    comparison_res_list = pickle.load(handle)

pvalue_estimation= dict()
pvalue_estimation['beta'] = [None]*iteration
pvalue_estimation['cross'] = dict()
pvalue_estimation['naive'] = dict()
pvalue_estimation['cross']['marginal_emp'] = [None]*iteration
pvalue_estimation['cross']['marginal_fitted'] =  [None]*iteration
pvalue_estimation['cross']['partial_emp'] =  [None]*iteration
pvalue_estimation['cross']['partial_fitted'] =  [None]*iteration

pvalue_estimation['naive']['marginal_emp'] =  [None]*iteration
pvalue_estimation['naive']['marginal_fitted'] =  [None]*iteration
pvalue_estimation['naive']['partial_emp'] =  [None]*iteration
pvalue_estimation['naive']['partial_fitted'] =  [None]*iteration

for it in np.arange(iteration):
    print("##############"+str(it)+"####################")
    file = curpath+'/data/data_subj' + str(I) + '_time' + str(T) + '_feature' + str(J) + \
           '_Z' + str(q) + '_rate' + str(rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
           str(SNR2) + '_iteration' + str(it) + '.npz'
    data = np.load(file)
    rank = 3; max_iter = 30
    data_obj = dict(X=data['Xobs'], O=data['Obs'], Z=data['Z'],
                    time_stamps=data['T0'],
                    rank=rank)
    pvalue_estimation['beta'][it] = data['beta0'].copy()
    tmp_obj = spaco.SPACOcv(data_obj)
    tmp_obj.train_preparation(run_prepare=True,
                      run_init=False,
                      mean_trend_removal=False,
                      smooth_penalty=False)
    comparison_res_list[it].spaco_fit.intermediantes= copy.deepcopy(tmp_obj.intermediantes)
    comparison_res_list[it].spaco_fit.O = copy.deepcopy(tmp_obj.O)
    comparison_res_list[it].spaco_fit.X = copy.deepcopy(tmp_obj.X)
    comparison_res_list[it].spaco_fit.X0 = copy.deepcopy(tmp_obj.X0)
    train_ids, test_ids = spaco.cutfoldid(n=I, nfolds=5,
                                          random_state=2022)
    cv_iter = 10
    comparison_res_list[it].spaco_fit.cross_validation_train(train_ids,
                                     test_ids,
                                     max_iter=cv_iter,
                                     min_iter=5,
                                     tol=1e-3,
                                     trace=True)
    spaco_fit = copy.deepcopy(comparison_res_list[it].spaco_fit)
    delta = 0;
    tol = 0.01;
    fixbeta0 = False;
    nfolds = 5
    random_state = 0
    for method in ["cross", "naive"]:
        feature_eval = spaco.CRtest_cross(spaco_fit,
                                      type=method,
                                      delta=delta)
        # precalculate quantities that stay the same for different Z
        feature_eval.precalculation()
        feature_eval.cut_folds(nfolds=nfolds,
                               random_state=random_state)
        feature_eval.beta_fun_full(nfolds=nfolds, max_iter=1,
                                   tol=tol, fixbeta0=fixbeta0)
        # drop each feature and refit
        for j in np.arange(feature_eval.Z.shape[1]):
            feature_eval.beta_fun_one(nfolds=nfolds, j=j,
                                      max_iter=1,
                                      fixbeta0=fixbeta0)
        # calculate the test statistics for conditional and mariginal independence
        for j in np.arange(feature_eval.Z.shape[1]):
            feature_eval.precalculation_response(j=j)
            feature_eval.coef_partial_fun(j=j, inplace=True)
            feature_eval.coef_marginal_fun(j=j, inplace=True)

        ##Generate randomized variables
        B = 200
        Zconditional = np.random.normal(size=(I, q, B))
        Zmarginal = Zconditional
        feature_eval.coef_partial_random = np.zeros((feature_eval.coef_partial.shape[0],
                                                     feature_eval.coef_partial.shape[1],
                                                     B))
        feature_eval.coef_marginal_random = np.zeros((feature_eval.coef_partial.shape[0],
                                                     feature_eval.coef_partial.shape[1],
                                                     B))

        for j in np.arange(feature_eval.Z.shape[1]):
            feature_eval.coef_random_fun(Zconditional, j, type = "partial")
            feature_eval.coef_random_fun(Zmarginal, j, type="marginal")


        pvals_partial_empirical, pvals_partial_fitted = \
            feature_eval.pvalue_calculation(type="partial",
                                            pval_fit=True,
                                            dist_name='nct')
        pvals_marginal_empirical, pvals_marginal_fitted = \
        feature_eval.pvalue_calculation(type = "marginal",pval_fit = True, dist_name ='nct')
        pvalue_estimation[method]['partial_emp'][it] = (pvals_partial_empirical.copy())
        pvalue_estimation[method]['partial_fitted'][it] = (pvals_partial_fitted.copy())
        pvalue_estimation[method]['marginal_emp'][it] = (pvals_marginal_empirical.copy())
        pvalue_estimation[method]['marginal_fitted'][it] = (pvals_marginal_fitted.copy())


with open(pvalue_file_name, 'wb') as handle:
    pickle.dump(pvalue_estimation, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

methods=["cross", "naive"]
types = ["partial_fitted", "marginal_fitted", "partial_emp", "marginal_emp"]
nz_beta = dict()
nz_pval = dict()
null_pval = dict()
for type in types:
    nz_beta[type] = dict()
    nz_pval[type] = dict()
    null_pval[type] = dict()

for type in types:
    for method in methods:
        nz_beta[type][method] = []
        nz_pval[type][method] = []
        null_pval[type][method] = []

for it in np.arange(iteration):
    for method in methods:
        for type in types:
            aa1 = pvalue_estimation['beta'][it][:3, :]
            idx = np.where(aa1 != 0)
            if len(idx[0]) > 0:
                tmp1 = list(aa1[idx])
                tmp2 = list(pvalue_estimation[method][type][it][:3,:][idx])
                tmp3 = list(pvalue_estimation[method][type][it][2:,:].reshape(-1))
                nz_beta[type][method]= nz_beta[type][method]+tmp1.copy()
                nz_pval[type][method] = nz_pval[type][method] +tmp2.copy()
                null_pval[type][method] = null_pval[type][method]+tmp3.copy()

for type in types:
    print("##########"+type+"###########")
    for method in methods:
        print(method)
        if len(nz_beta[type][method]) > 0:
            tmp_mat = np.zeros((len(nz_beta[type][method]),2))
            tmp_mat[:,0] = nz_beta[type][method]
            tmp_mat[:,1] = nz_pval[type][method]
            print(tmp_mat)
        qs = np.arange(1,10)/100.0
        a = null_pval[type][method]
        print(np.quantile(a, qs))


