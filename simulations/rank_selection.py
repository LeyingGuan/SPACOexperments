import spaco as spaco
import numpy as np
import sys
import os
import pickle
import importlib
importlib.reload(spaco)

#I = 100; T = 30; q = 100;J = 10;SNR1 = 3.0;SNR2 = 3.0;rate = 0.5
I = int(sys.argv[1])
T = int(sys.argv[2])
J = int(sys.argv[3])
q = int(sys.argv[4])
rate = float(sys.argv[5])
SNR1 = float(sys.argv[6])
SNR2 = float(sys.argv[7])
iteration = 20
curpath = "/home/lg689/project/SPACOresults"

res_file_name = curpath + '/results/rank_selection/data_subj' + str(
    I) + '_time' + str(T) + '_feature' + str(J) + \
                '_Z' + str(q) + '_rate' + str(
    rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
                str(SNR2) + '.pickle'

rescontruction_errs = np.zeros((iteration, 2)) * np.nan
selected_ranks_collection = np.zeros((iteration,1)) * np.nan
for it in np.arange(iteration):
    print("##############"+str(it)+"####################")
    file = curpath+'/data/data_subj' + str(I) + '_time' + str(T) + '_feature' + str(J) + \
           '_Z' + str(q) + '_rate' + str(rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
           str(SNR2) + '_iteration' + str(it) + '.npz'
    data = np.load(file)
    max_iter = 30
    true_rank = 3
    ranks = np.arange(1, 11)
    negliks = spaco.rank_selection_function(X=data['Xobs'], O=data['Obs'], Z=data['Z'],
                                  time_stamps=data['T0'],
                                  ranks=ranks,
                                  early_stop=True,
                                  max_iter=30, cv_iter=5)
    means = negliks.mean(axis=0)
    means=means[~np.isnan(means)]
    idx_min = np.argmin(means)
    rank_min = ranks[idx_min]
    selected_ranks_collection[it,0] = rank_min

    data_obj = dict(X=data['Xobs'], O=data['Obs'], Z=data['Z'], time_stamps=data['T0'],
                    rank=true_rank)
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
    eval = spaco.metrics(U=spaco_fit.mu, Phi=spaco_fit.Phi, V=spaco_fit.V,
                 U0=data['U'], Phi0 = data['Phi0'], V0 = data['V0'])
    eval.reconstruction_error(O = data['Obs'])
    rescontruction_errs[it, 0] =eval.alignment['reconstruct_cor_full']
    data_obj['rank'] =rank_min
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
    eval = spaco.metrics(U=spaco_fit.mu, Phi=spaco_fit.Phi, V=spaco_fit.V,
                 U0=data['U'], Phi0 = data['Phi0'], V0 = data['V0'])
    eval.reconstruction_error(O = data['Obs'])
    rescontruction_errs[it, 1] =eval.alignment['reconstruct_cor_full']

rank_eval = dict()
rank_eval['selected_rank'] = selected_ranks_collection
rank_eval['reconstr_err'] = rescontruction_errs

print(rank_eval)

with open(res_file_name, 'wb') as handle:
    pickle.dump(rank_eval, handle,
                protocol=pickle.HIGHEST_PROTOCOL)


# with open(res_file_name, 'rb') as handle:
#     b = pickle.load(handle)






