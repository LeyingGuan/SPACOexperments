import sys
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib import patches as mpatches

I = 100; T = 30; q = 100;
Js=[10, 500]
rates=[1.0, 0.5, 0.1]
SNR1s=[1.0, 3.0, 5.0]
SNR2s = [0.0, 3.0, 10.0]

#curpath = "/Users/lg689/Desktop/SPACOresults"
curpath = '/home/lg689/project/SPACOresults'
rank_mats = None
cor_mats = None
for J in Js:
    for rate in rates:
        for SNR1 in SNR1s:
            for SNR2 in SNR2s:
                res_file_name = curpath + '/results/rank_selection/data_subj' + str(
                    I) + '_time' + str(T) + '_feature' + str(
                    J) +  '_Z' + str(q) + '_rate' + str(rate) + '_SNR1' + str(SNR1) + \
                                '_SNR2' + str(SNR2) + '.pickle'
                file_exists = exists(res_file_name)
                if file_exists:
                    print(res_file_name)
                    with open(res_file_name, 'rb') as handle:
                        rank_res = pickle.load( handle)
                    tmp_rank_mats = np.zeros((rank_res['selected_rank'].shape[0], 5))
                    tmp_cor_mats = np.zeros((rank_res['selected_rank'].shape[0]*2, 6))
                    tmp_rank_mats[:,0] = rank_res['selected_rank'][:,0].copy()
                    tmp_rank_mats[:, 1] = SNR1
                    tmp_rank_mats[:,2] = J
                    tmp_rank_mats[:,3] = rate
                    tmp_rank_mats[:, 4] = SNR2
                    tmp_cor_mats[:rank_res['selected_rank'].shape[0],0] = rank_res['reconstr_err'][:,0]
                    tmp_cor_mats[:rank_res['selected_rank'].shape[0],1] = 0
                    tmp_cor_mats[rank_res['selected_rank'].shape[0]:,0] = rank_res['reconstr_err'][:,1]
                    tmp_cor_mats[rank_res['selected_rank'].shape[0]:,1] = 1
                    tmp_cor_mats[:,2] = SNR1
                    tmp_cor_mats[:, 3] = J
                    tmp_cor_mats[:, 4] = rate
                    tmp_cor_mats[:, 5] = SNR2
                    if rank_mats is None:
                        rank_mats = tmp_rank_mats.copy()
                        cor_mats = tmp_cor_mats.copy()
                    else:
                        rank_mats= np.vstack([rank_mats, tmp_rank_mats.copy()])
                        cor_mats = np.vstack([cor_mats,tmp_cor_mats.copy()])

rank_mats  = pd.DataFrame(rank_mats, columns= ["rank", "SNR1", "J", "rate", "SNR2"])
cor_mats = pd.DataFrame(cor_mats, columns= ["corr", "rank" ,"SNR1", "J", "rate","SNR2"])
my_pal = {"10": "steelblue", "500": "cornflowerblue"}
rank_mats['J'] =rank_mats['J'].values.astype('int').astype('str')
cor_mats['J'] =cor_mats['J'].values.astype('int').astype('str')
cor_mats['rank'] =cor_mats['rank'].values.astype('int').astype('str')
cor_mats['rank'][cor_mats['rank']=="1"] = "estimated"
cor_mats['rank'][cor_mats['rank']=="0"] = "correct"
fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex= True)
for iSNR1 in range(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    tmpDF1 = rank_mats[(rank_mats["SNR1"] == SNR1)&(rank_mats["J"] == "10")]['rank']
    tmpDF2 = rank_mats[(rank_mats["SNR1"] == SNR1) & (rank_mats["J"] == "500")]['rank']
    tmpDF1 = tmpDF1.value_counts()
    tmpDF2 = tmpDF2.value_counts()
    tmpDF1 = tmpDF1/sum(tmpDF1)
    tmpDF2 = tmpDF2/sum(tmpDF2)
    tmpDF1 = pd.DataFrame({"J": ["10"] * len(tmpDF1), "freq":tmpDF1, "rank":tmpDF1.index})
    tmpDF2 = pd.DataFrame({"J": ["500"] * len(tmpDF2), "freq": tmpDF2,"rank": tmpDF2.index})
    tmpDF = pd.concat([tmpDF1, tmpDF2])
    g = sns.barplot(ax = axes1[iSNR1], x="rank", y="freq", hue="J", data=tmpDF, palette=my_pal)
    axes1[iSNR1].axvline(2.0, color='red')
    axes1[iSNR1].set_ylim([0, 1.0])
    axes1[iSNR1].set_ylabel("freq(SNR1=" + str(int(SNR1))+")")
    g.legend_.remove()

J10 = mpatches.Patch(color="steelblue")
J500 = mpatches.Patch(color="cornflowerblue")
sp_name = ['J=10', 'J=500']
fig1.legend(title=None, labels=sp_name,
            handles=[J10, J500],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})

my_pal2 = {"correct": "steelblue", "estimated": "coral"}
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex= True)
for iSNR1 in range(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    tmpDF = cor_mats[(cor_mats["SNR1"] == SNR1)]
    g = sns.barplot(ax = axes2[iSNR1], x="J", y="corr", hue="rank", data=tmpDF, palette=my_pal2)
    axes2[iSNR1].set_ylim([0, 1.0])
    axes2[iSNR1].set_ylabel("corr(SNR1=" + str(int(SNR1))+")")
    g.legend_.remove()

correct = mpatches.Patch(color="steelblue")
estimated = mpatches.Patch(color="coral")
sp_name = ['correct rank', 'estimated rank']
fig2.legend(title=None, labels=sp_name,
            handles=[correct, estimated],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})

#fig.show()
FIG_file1 =  curpath + '/results/rank_selection.png'
fig1.savefig(FIG_file1)
plt.close(fig1)

FIG_file2 =  curpath + '/results/reconstruction_selectedrank.png'
fig2.savefig(FIG_file2)
plt.close(fig2)
