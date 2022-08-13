import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib import patches as mpatches
import numpy as np
import pandas as pd
#evaluation construction
#FIG1: subfigure: color = methods, x = missing rate * J, SNR1 \by SNR2
#Supp FIG1: same, but on the missing values only
#Supp FIG2: different initializations
#FIG2: subspace alignment of Phi: SubCP, SPACO
#FIG3: subspace alignment of U: SubCP, SPACO
Js=[10, 500]
rates=[1.0, 0.5, 0.1]
SNR1s=[1.0, 3.0, 5.0]
SNR2s=[0.0, 3.0, 10.0]

I = 100; T = 30; q = 100; iteration = 20
#J = 10;SNR1 = 1.0;SNR2 = 10.0;rate = 0.5
curpath = '/home/lg689/project/SPACOresults'
Js=[10, 500]
rates=[1.0, 0.5, 0.1]
SNR1s=[1.0, 3.0, 5.0]
SNR2s=[0.0, 3.0, 10.0]
methods=["cross", "naive"]
types = ["partial_fitted", "marginal_fitted", "partial_emp", "marginal_emp"]
nz_beta = dict()
nz_pval = dict()
null_pval = dict()

for J in Js:
    for rate in rates:
        for SNR1 in SNR1s:
            for SNR2 in SNR2s:
                idx_name = "J"+str(J)+"_r"+str(rate)+"_SNR1"+str(SNR1)+"_SNR2"+str(SNR2)
                nz_beta[idx_name] = dict()
                nz_pval[idx_name] = dict()
                null_pval[idx_name] = dict()
                for type in types:
                    nz_beta[idx_name][type] = dict()
                    nz_pval[idx_name][type] = dict()
                    null_pval[idx_name][type] = dict()
                for type in types:
                    for method in methods:
                        nz_beta[idx_name][type][method] = []
                        nz_pval[idx_name][type][method] = []
                        null_pval[idx_name][type][method] = []
                pvalue_file_name = curpath + '/results/test/data_subj' + str(
                    I) + '_time' + str(T) + '_feature' + str(J) + \
                                '_Z' + str(q) + '_rate' + str(
                    rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
                                str(SNR2) + '.pickle'

                with open(pvalue_file_name, 'rb') as handle:
                    pvalue_estimation = pickle.load(handle)
                for it in np.arange(iteration):
                    for method in methods:
                        for type in types:
                            tmp3 = list(pvalue_estimation[method][type][it][3:,:].reshape(-1))
                            null_pval[idx_name][type][method] = null_pval[idx_name][type][method] + tmp3.copy()
                            aa1 = pvalue_estimation['beta'][it][:3, :]
                            idx = np.where(aa1 != 0)
                            if len(idx[0]) > 0:
                                tmp1 = list(aa1[idx])
                                tmp2 = list(pvalue_estimation[method][type][it][:3,:][idx])
                                nz_beta[idx_name][type][method]= nz_beta[idx_name][type][method]+tmp1.copy()
                                nz_pval[idx_name][type][method] = nz_pval[idx_name][type][method] +tmp2.copy()

Js=[10, 500]
rates=[1.0, 0.5, 0.1]
SNR1s=[1.0, 3.0, 5.0]
SNR2s=[0.0, 3.0, 10.0]
alphas = [0.01, 0.05]
methods=["cross", "naive"]
types = ["partial_fitted", "marginal_fitted", "partial_emp", "marginal_emp"]
typeI_power = dict()
for type0 in types:
    typeI_power[type0] = dict()
    for method in methods:
        print(method)
        typeI_power[type0][method] = dict()
        for J in Js:
            for rate in rates:
                for SNR1 in SNR1s:
                    for SNR2 in SNR2s:
                        idx_name = "J" + str(J) + "_r" + str(rate) + "_SNR1" + str(SNR1) + "_SNR2" + str(SNR2)
                        typeI_power[type0][method][idx_name] = np.zeros((len(alphas), 2))
                        typeI_power[type0][method][idx_name][:] = np.nan
                        for ialpha in np.arange(len(alphas)):
                            typeI_power[type0][method][idx_name][ialpha, 0] = np.mean(np.array(null_pval[idx_name][type0][method])<alphas[ialpha])
                            if len(nz_beta[idx_name][type0][method]) > 0:
                                typeI_power[type0][method][idx_name][ialpha,1] = np.mean(np.array(nz_pval[idx_name][type0][method])<alphas[ialpha])

typeI_power
###
types = ["partial_fitted", "marginal_fitted"]

typeI_power_visualization = dict()
typeI_power_visualization["type"] = []
typeI_power_visualization["J"] = []
typeI_power_visualization["rate"] = []
typeI_power_visualization["SNR1"] = []
typeI_power_visualization["SNR2"] = []
typeI_power_visualization["alpha"] = []
typeI_power_visualization["typeI"] = []
typeI_power_visualization["power"] = []
for type0 in types:
    for J in Js:
        for rate in rates:
            for SNR1 in SNR1s:
                for SNR2 in SNR2s:
                    idx_name = "J" + str(J) + "_r" + str(rate) + "_SNR1" + str(SNR1) + "_SNR2" + str(SNR2)
                    for method in ['cross']:
                        for ialpha in np.arange(len(alphas)):
                            if type0 == 'partial_fitted':
                                typeI_power_visualization["type"].append('partial')
                            if type0 == 'marginal_fitted':
                                typeI_power_visualization["type"].append('marginal')
                            typeI_power_visualization["J"].append(J)
                            typeI_power_visualization["rate"].append(rate)
                            typeI_power_visualization["SNR1"].append(SNR1)
                            typeI_power_visualization["SNR2"].append(SNR2)
                            typeI_power_visualization["alpha"].append(alphas[ialpha])
                            typeI_power_visualization["typeI"].append(typeI_power[type0][method][idx_name][ialpha,0])
                            typeI_power_visualization["power"].append(typeI_power[type0][method][idx_name][ialpha,1])

typeI_power_visualizationDF = pd.DataFrame.from_dict(typeI_power_visualization)

my_pal = {"partial": "coral", "marginal": "steelblue"}
partial = mpatches.Patch(color="coral")
marginal = mpatches.Patch(color="steelblue")
sp_name = ['partial', 'marginal']

for rate in rates:
    typeI_power_visualizationDF1 = typeI_power_visualizationDF[(typeI_power_visualizationDF['rate']==rate)].copy()
    fig, axes = plt.subplots(3, 3, figsize=(9, 12), sharey=False, sharex= True)
    for iSNR1 in np.arange(len(SNR1s)):
        SNR1 = SNR1s[iSNR1]
        for iSNR2 in np.arange(len(SNR2s)):
            SNR2 = SNR2s[iSNR2]
            typeI_power_visualizationDF2 = typeI_power_visualizationDF1[(typeI_power_visualizationDF['SNR1']==SNR1) & (typeI_power_visualizationDF['SNR2']==SNR2)]
            typeI_power_visualizationDF2['xname'] = ['alpha'+str(typeI_power_visualizationDF2["alpha"].iloc[i])+"_J"+str(typeI_power_visualizationDF2["J"].iloc[i]) for i in np.arange(typeI_power_visualizationDF2.shape[0])]
            g1 = sns.barplot(x="xname", y="typeI",
                        hue="type", data=typeI_power_visualizationDF2,
                             palette=my_pal,
                        ci=None, ax = axes[iSNR1, iSNR2])
            axes[iSNR1, iSNR2].set_ylim([0, 0.1])
            g1.axhline(0.01, color="lightgray", linestyle="--")
            g1.axhline(0.05, color="lightgray", linestyle="--")
            if iSNR2 != 0:
                axes[iSNR1, iSNR2].get_yaxis().set_ticklabels([])
                axes[iSNR1, iSNR2].set_ylabel(None)
            else:
                axes[iSNR1, iSNR2].set_ylabel("SNR1"+str(SNR1))
            if iSNR1!=2:
                axes[iSNR1, iSNR2].get_xaxis().set_ticklabels([])
                axes[iSNR1, iSNR2].set_xlabel(None)
            else:
                axes[iSNR1, iSNR2].tick_params(axis='x',rotation=90)
                axes[iSNR1, iSNR2].set_xlabel('SNR2'+str(SNR2))
            g1.legend_.remove()
    fig.legend(title=None, labels=sp_name,
                handles=[partial, marginal],
                fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
               prop={'size': 12})
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(bottom=0.2)
    FIG_file1 = curpath + '/results/typeI_r'+str(rate)+'.png'
    fig.savefig(FIG_file1)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(9, 12),sharey=False, sharex=True)
    for iSNR1 in np.arange(len(SNR1s)):
        SNR1 = SNR1s[iSNR1]
        for iSNR2 in np.arange(1,len(SNR2s)):
            SNR2 = SNR2s[iSNR2]
            typeI_power_visualizationDF2 = typeI_power_visualizationDF1[(typeI_power_visualizationDF['SNR1']==SNR1) & (typeI_power_visualizationDF['SNR2']==SNR2)]
            typeI_power_visualizationDF2['xname'] = ['alpha'+str(typeI_power_visualizationDF2["alpha"].iloc[i])+"_J"+str(typeI_power_visualizationDF2["J"].iloc[i]) for i in np.arange(typeI_power_visualizationDF2.shape[0])]
            g2 = sns.barplot(x="xname", y="power",
                        hue="type", data=typeI_power_visualizationDF2,
                             palette=my_pal,
                        ci=None, ax = axes[iSNR1, iSNR2-1])
            if iSNR2 != 1:
                axes[iSNR1, iSNR2-1].get_yaxis().set_ticklabels([])
                axes[iSNR1, iSNR2-1].set_ylabel(None)
            else:
                axes[iSNR1, iSNR2-1].set_ylabel("SNR1" + str(SNR1))
            if iSNR1!=2:
                axes[iSNR1, iSNR2-1].get_xaxis().set_ticklabels([])
                axes[iSNR1, iSNR2-1].set_xlabel(None)
            else:
                axes[iSNR1, iSNR2-1].tick_params(axis='x',rotation=90)
                axes[iSNR1, iSNR2-1].set_xlabel('SNR2'+str(SNR2))
            axes[iSNR1, iSNR2-1].set_ylim(0, 1.0)
            g2.legend_.remove()
    fig.legend(title=None, labels=sp_name,
                handles=[partial, marginal],
                fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
               prop={'size': 12})
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.subplots_adjust(bottom=0.2)
    FIG_file1 = curpath + '/results/power_r'+str(rate)+'.png'
    fig.savefig(FIG_file1)
    plt.close(fig)

Js=[10, 500]
rates=[1.0, 0.5, 0.1]
SNR1s=[1.0, 3.0, 5.0]
SNR2s=[0.0, 3.0, 10.0]
types = ["partial_fitted", "marginal_fitted", "partial_emp", "marginal_emp"]
type = "partial_fitted"
eps = 1e-6
for type in types:
    for J in Js:
        for rate in rates:
            rows = 3;
            cols = 3;
            figsize = (9, 9)
            figs_null, axis_null = plt.subplots(nrows=rows,
                                                ncols=cols,
                                                sharex=False,
                                                sharey=False,
                                                figsize=figsize)
            plt.subplots_adjust(left=0.6 / figsize[0],
                                bottom=0.6 / figsize[0],
                                right=1.0 - 0.5 / figsize[0],
                                top=1.0 - .8 / figsize[1])
            for iSNR1 in np.arange(len(SNR1s)):
                for iSNR2 in np.arange(len(SNR1s)):
                    SNR1 = SNR1s[iSNR1]
                    SNR2 = SNR2s[iSNR2]
                    idx_name = "J" + str(J) + "_r" + str(rate) + "_SNR1" + str(SNR1) + "_SNR2" + str(SNR2)
                    print(idx_name)
                    pval_null_cross = np.array(null_pval[idx_name][type]['cross'])
                    pval_null_naive = np.array(null_pval[idx_name][type]['naive'])
                    pval_null_cross = pval_null_cross[~np.isnan(pval_null_cross)]
                    pval_null_naive = pval_null_naive[~np.isnan(pval_null_naive)]
                    pval_null_cross = -np.log10(-np.sort(-pval_null_cross)+eps)
                    pval_null_naive =  -np.log10(-np.sort(-pval_null_naive)+eps)
                    pval_null1_cross = (np.arange(len(pval_null_cross)) + 0.5) / len(pval_null_cross)
                    pval_null1_naive = (np.arange(len(pval_null_naive)) + 0.5) / len(pval_null_naive)
                    pval_null1_cross = -np.log10(-np.sort(-pval_null1_cross)+eps)
                    pval_null1_naive= -np.log10(-np.sort(-pval_null1_naive)+eps)
                    pval_all = np.vstack([pval_null_cross, pval_null1_cross, pval_null1_naive, pval_null_naive]).reshape(-1)
                    pval_all = np.sort(pval_all)
                    alpha_trans = 0.5
                    axis_null[iSNR1 , iSNR2].plot(pval_all, pval_all, color='b',linestyle='-')
                    axis_null[iSNR1 , iSNR2].scatter(pval_null1_cross,pval_null_cross, s=2,color='r', alpha=alpha_trans )
                    axis_null[iSNR1 , iSNR2].scatter(pval_null1_naive,pval_null_naive, s=2,color='b', alpha=alpha_trans)
                    if iSNR2 == 0:
                        axis_null[iSNR1, iSNR2].set_ylabel(
                            "SNR1=" + str(int(SNR1)))
                    if iSNR2 != 0:
                        # axes[iSNR1,iSNR2].get_yaxis().set_visible(False)
                        #axis_null[iSNR1, iSNR2].get_yaxis().set_ticklabels([])
                        axis_null[iSNR1, iSNR2].set_ylabel(None)
                    if iSNR1 != 2:
                        #axis_null[iSNR1, iSNR2].get_xaxis().set_ticklabels([])
                        axis_null[iSNR1, iSNR2].set_xlabel(None)
                    if iSNR1 == 2:
                        axis_null[iSNR1, iSNR2].tick_params(axis='x',
                                                       rotation=75)
                        axis_null[iSNR1, iSNR2].set_xlabel("SNR2=" + str(int(SNR2)))
            FIG_file1 = curpath + '/results/'+type+'_qqplot_J'+str(J)+"_r"+str(rate)+'.png'
            figs_null.suptitle(type +",J="+str(J)+', observing rate='+str(rate))
            figs_null.savefig(FIG_file1)
            #figs_null.show()
            plt.close(figs_null)

