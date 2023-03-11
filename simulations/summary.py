import sys
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib import patches as mpatches
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

I = 100; T = 30; q = 100;
#J = 10;SNR1 = 1.0;SNR2 = 10.0;rate = 0.5
curpath = '/home/lg689/project/SPACOresults'

construction_cor_list = dict(spaco = [],
                             spaco_=[],
                             SupCP_functional=[],
                             SupCP_random=[],
                             cp=[],
                             Raw = [],
                             SNR1 = [],
                             SNR2 = [],
                             rate = [],
                             J = [])
construction_cor_list_miss = dict(spaco = [],
                                  spaco_=[],
                                  SupCP_functional=[],
                                  SupCP_random=[],
                                  cp=[],
                                  Raw = [],
                                  SNR1=[],
                                  SNR2=[],
                                  rate=[],
                                  J=[]
                                  )

U_align_list = dict(spaco = [],
                      spaco_=[],
                      SupCP_functional=[],
                      rank = [],
                      SNR1=[],
                      SNR2=[],
                      rate=[],
                      J=[]
                      )

Phi_align_list = dict(spaco = [],
                      spaco_=[],
                      SupCP_functional=[],
                      rank = [],
                      SNR1=[],
                      SNR2=[],
                      rate=[],
                      J=[]
                      )
method_names = ['spaco', 'spaco_', 'SupCP_functional','SupCP_random','cp']
method_names1 = ['spaco', 'spaco_', 'SupCP_functional']
for J in Js:
    for rate in rates:
        for SNR1 in SNR1s:
            for SNR2 in SNR2s:
                res_file_name = curpath + '/results/construction/data_subj' + str(
                    I) + '_time' + str(
                    T) + '_feature' + str(J) + \
                                '_Z' + str(
                    q) + '_rate' + str(
                    rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
                                str(SNR2) + '.pickle'
                file_exists = exists(res_file_name)
                if file_exists:
                    print(res_file_name)
                    with open(res_file_name, 'rb') as handle:
                        comparison_res_list = pickle.load(handle)
                    for it in np.arange(len(comparison_res_list)):
                        comparison_res = comparison_res_list[it]
                        construction_cor_list['Raw'].append(comparison_res.eval_dict['empirical']['reconstruct_cor_full'])
                        construction_cor_list_miss['Raw'].append(comparison_res.eval_dict['empirical']['reconstruct_cor_miss'])
                        construction_cor_list_miss['SNR1'].append(SNR1)
                        construction_cor_list_miss['SNR2'].append(SNR2)
                        construction_cor_list_miss['J'].append(J)
                        construction_cor_list_miss['rate'].append(rate)
                        construction_cor_list['SNR1'].append(SNR1)
                        construction_cor_list[ 'SNR2'].append(SNR2)
                        construction_cor_list['J'].append(J)
                        construction_cor_list[ 'rate'].append(rate)
                        Phi_align_list['SNR1']=Phi_align_list['SNR1']+([SNR1]*3)
                        Phi_align_list[ 'SNR2']= Phi_align_list[ 'SNR2']+([SNR2]*3)
                        Phi_align_list['J'] = Phi_align_list['J']+[J]*3
                        Phi_align_list[ 'rate'] = Phi_align_list[ 'rate']+([rate]*3)
                        Phi_align_list['rank'] = Phi_align_list['rank']+[0, 1, 2]
                        U_align_list['SNR1']=U_align_list['SNR1']+([SNR1]*3)
                        U_align_list[ 'SNR2']= U_align_list[ 'SNR2']+([SNR2]*3)
                        U_align_list['J'] = U_align_list['J']+[J]*3
                        U_align_list[ 'rate'] = U_align_list[ 'rate']+([rate]*3)
                        U_align_list['rank'] = U_align_list['rank']+[0, 1, 2]
                        for method in method_names:
                            if method in comparison_res.eval_dict.keys():
                                construction_cor_list[method].append(comparison_res.eval_dict[method].alignment['reconstruct_cor_full'])
                                construction_cor_list_miss[method].append(comparison_res.eval_dict[method].alignment['reconstruct_cor_miss'])
                            else:
                                construction_cor_list[method].append(np.nan)
                                construction_cor_list_miss[method].append(np.nan)
                        for method in method_names1:
                            if method in comparison_res.eval_dict.keys():
                                Phi_align_list[method].append(comparison_res.eval_dict[method].alignment['Phi'][0])
                                Phi_align_list[method].append(comparison_res.eval_dict[method].alignment['Phi'][1])
                                Phi_align_list[method].append(comparison_res.eval_dict[method].alignment['Phi'][2])
                                U_align_list[method].append(comparison_res.eval_dict[method].alignment['U'][0])
                                U_align_list[method].append(comparison_res.eval_dict[method].alignment['U'][1])
                                U_align_list[method].append(comparison_res.eval_dict[method].alignment['U'][2])
                            else:
                                Phi_align_list[method] = Phi_align_list[method]+([np.nan]*3)
                                U_align_list[method] = U_align_list[method] + ([np.nan] * 3)



construction_corDF = pd.DataFrame.from_dict(construction_cor_list)
construction_corMissDF = pd.DataFrame.from_dict(construction_cor_list_miss)
Phi_align_DF = pd.DataFrame.from_dict(Phi_align_list)
U_align_DF = pd.DataFrame.from_dict(U_align_list)

method_names_plot = ['spaco', 'spaco_', 'SupCP_functional','cp']
construction_corDF1 = pd.melt(construction_corDF,
        id_vars= ["J", "SNR1", "SNR2", "rate"],
        value_vars= method_names_plot+["Raw"],
                              var_name='method', value_name="correlation")

construction_corDF1=construction_corDF1[~np.isnan(construction_corDF1['correlation'])]
sns.set_context("notebook", font_scale=1.2)
#sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False, sharex= True)
A = construction_corDF1[["J","SNR1","SNR2","rate"]]
A.drop_duplicates()

A = construction_corDF1[["J","method"]]
A.drop_duplicates()

my_pal = {"spaco": "darkred", "spaco_": "coral", "SupCP_functional": "steelblue",
          "SupCP_random":"cornflowerblue", "cp":"darkcyan",
          "Raw":"lightgray"}

spaco = mpatches.Patch(color="darkred")
spaco_ = mpatches.Patch(color="coral")
SupCP_functional = mpatches.Patch(color= "steelblue")
SupCP_random=mpatches.Patch(color= "cornflowerblue")
cp =mpatches.Patch(color= "darkcyan")
Raw=mpatches.Patch(color= "lightgray")
#sp_name = ['SPACO', 'SPACO-', 'SupCP', 'SupCP_rd', "CP", "Raw"]
sp_name = ['SPACO', 'SPACO-', 'SupCP', "CP", "Raw"]

for iSNR1 in np.arange(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    #tmpDF1 = construction_corDF1[(construction_corDF1["J"] == J) & (construction_corDF1["SNR1"] == SNR1)]
    tmpDF1 = construction_corDF1[(construction_corDF1["SNR1"] == SNR1)]
    ymax = tmpDF1["correlation"].max()+0.1
    if ymax > 1:
        ymax = 1
    ymin = tmpDF1["correlation"].min()-0.05
    for iSNR2 in np.arange(len(SNR2s)):
        SNR2 = SNR2s[iSNR2]
        #tmpDF = construction_corDF1[(construction_corDF1["J"]==J)&(construction_corDF1["SNR1"]==SNR1)&(construction_corDF1["SNR2"]==SNR2)]
        tmpDF = construction_corDF1[(construction_corDF1["SNR1"] == SNR1) & (construction_corDF1["SNR2"] == SNR2)]
        tmpDF["J_rate"] = ['J'+str(tmpDF["J"].iloc[i])+"_r"+str(str(tmpDF["rate"].iloc[i])) for i in np.arange(tmpDF.shape[0])]
        g = sns.boxplot(ax = axes[iSNR1,iSNR2],x="J_rate",
                    y='correlation',
                    hue='method',
                    palette=my_pal,
                    data=tmpDF);
        axes[iSNR1, iSNR2].set_ylim([ymin, ymax])
        if iSNR2 == 0:
            axes[iSNR1, iSNR2].set_ylabel("SNR1="+str(int(SNR1)))
        if iSNR2 != 0:
            #axes[iSNR1,iSNR2].get_yaxis().set_visible(False)
            axes[iSNR1,iSNR2].get_yaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_ylabel(None)
        if iSNR1 != 2:
            axes[iSNR1,iSNR2].get_xaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_xlabel(None)
        if iSNR1 == 2:
            axes[iSNR1, iSNR2].tick_params(axis='x', rotation=75)
            axes[iSNR1, iSNR2].set_xlabel("SNR2="+str(int(SNR2)))
        g.legend_.remove()


fig.legend(title=None, labels=sp_name,
            handles=[spaco, spaco_, SupCP_functional, cp, Raw],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})
fig.tight_layout()
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(hspace=0.2)
#fig.show()
FIG_file1 =  curpath + '/results/reconstruction_view_signalmat.png'
fig.savefig(FIG_file1)
plt.close(fig)


Phi_align_DF = pd.DataFrame.from_dict(Phi_align_list)

method_names_plot = ['spaco', 'SupCP_functional']
Phi_align_corDF1 = pd.melt(Phi_align_DF,
        id_vars= ["J", "rank","SNR1", "SNR2", "rate"],
        value_vars= method_names_plot,
                              var_name='method', value_name="correlation")

Phi_align_corDF1=Phi_align_corDF1[~np.isnan(Phi_align_corDF1['correlation'])]
Phi_align_corDF1=Phi_align_corDF1[Phi_align_corDF1['J']==10]

sns.set_context("notebook", font_scale=1.2)
#sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False, sharex= True)

my_pal = {"spaco": "darkred", "spaco_": "coral", "SupCP_functional": "steelblue",
          "SupCP_random":"cornflowerblue", "cp":"darkcyan",
          "Raw":"lightgray"}

spaco = mpatches.Patch(color="darkred")
spaco_ = mpatches.Patch(color="coral")
SupCP_functional = mpatches.Patch(color= "steelblue")
SupCP_random=mpatches.Patch(color= "cornflowerblue")
cp =mpatches.Patch(color= "darkcyan")
Raw=mpatches.Patch(color= "lightgray")
#sp_name = ['SPACO', 'SPACO-', 'SupCP', 'SupCP_rd', "CP", "Raw"]
sp_name = ['SPACO','SupCP']

for iSNR1 in np.arange(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    #tmpDF1 = construction_corDF1[(construction_corDF1["J"] == J) & (construction_corDF1["SNR1"] == SNR1)]
    tmpDF1 = Phi_align_corDF1[(Phi_align_corDF1["SNR1"] == SNR1)]
    ymax = tmpDF1["correlation"].max()+0.05
    if ymax > 1:
        ymax = 1
    ymin = tmpDF1["correlation"].min()-0.05
    if ymin < 0:
        ymin = 0
    for iSNR2 in np.arange(len(SNR2s)):
        SNR2 = SNR2s[iSNR2]
        #tmpDF = construction_corDF1[(construction_corDF1["J"]==J)&(construction_corDF1["SNR1"]==SNR1)&(construction_corDF1["SNR2"]==SNR2)]
        tmpDF = Phi_align_corDF1[(Phi_align_corDF1["SNR1"] == SNR1) & (Phi_align_corDF1["SNR2"] == SNR2)]
        tmpDF["comp_rate"] = ['C'+str(tmpDF["rank"].iloc[i]+1)+"_r"+str(str(tmpDF["rate"].iloc[i])) for i in np.arange(tmpDF.shape[0])]
        g = sns.boxplot(ax = axes[iSNR1,iSNR2],x="comp_rate",
                    y='correlation',
                    hue='method',
                    palette=my_pal,
                    data=tmpDF);
        axes[iSNR1, iSNR2].set_ylim([ymin, ymax])
        if iSNR2 == 0:
            axes[iSNR1, iSNR2].set_ylabel("SNR1="+str(int(SNR1)))
        if iSNR2 != 0:
            #axes[iSNR1,iSNR2].get_yaxis().set_visible(False)
            axes[iSNR1,iSNR2].get_yaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_ylabel(None)
        if iSNR1 != 2:
            axes[iSNR1,iSNR2].get_xaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_xlabel(None)
        if iSNR1 == 2:
            axes[iSNR1, iSNR2].tick_params(axis='x', rotation=75)
            axes[iSNR1, iSNR2].set_xlabel("SNR2="+str(int(SNR2)))
        g.legend_.remove()


fig.legend(title=None, labels=sp_name,
            handles=[spaco,   SupCP_functional],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})
fig.tight_layout()
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(hspace=0.2)
#fig.show()
FIG_file1 =  curpath + '/results/reconstruction_view_phi.png'
fig.savefig(FIG_file1)
plt.close(fig)


method_names_plot = ['spaco', 'spaco_']
Phi_align_corDF1 = pd.melt(U_align_DF,
        id_vars= ["J", "rank","SNR1", "SNR2", "rate"],
        value_vars= method_names_plot,
                              var_name='method', value_name="correlation")

Phi_align_corDF1=Phi_align_corDF1[~np.isnan(Phi_align_corDF1['correlation'])]
Phi_align_corDF1=Phi_align_corDF1[(Phi_align_corDF1['J']==10) & (Phi_align_corDF1['SNR2']==10)]

sns.set_context("notebook", font_scale=1.2)
#sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False, sharex= True)

my_pal = {"spaco": "darkred", "spaco_": "coral", "SupCP_functional": "steelblue",
          "SupCP_random":"cornflowerblue", "cp":"darkcyan",
          "Raw":"lightgray"}

spaco = mpatches.Patch(color="darkred")
spaco_ = mpatches.Patch(color="coral")
SupCP_functional = mpatches.Patch(color= "steelblue")
SupCP_random=mpatches.Patch(color= "cornflowerblue")
cp =mpatches.Patch(color= "darkcyan")
Raw=mpatches.Patch(color= "lightgray")
#sp_name = ['SPACO', 'SPACO-', 'SupCP', 'SupCP_rd', "CP", "Raw"]
sp_name = ['SPACO','SPACO-']
COMP = list(Phi_align_corDF1['rank'].unique())
ymaxs = [1.0, 0.5, 0.1]
for iSNR1 in np.arange(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    #tmpDF1 = construction_corDF1[(construction_corDF1["J"] == J) & (construction_corDF1["SNR1"] == SNR1)]
    tmpDF1 = Phi_align_corDF1[(Phi_align_corDF1["SNR1"] == SNR1)]
    #ymax = ymaxs[iSNR1]
    ymax = tmpDF1["correlation"].max() + 0.05
    ymin = tmpDF1["correlation"].min() - 0.05
    if ymin < 0:
        ymin = 0
    for iSNR2 in np.arange(len(COMP)):
        SNR2 = COMP[iSNR2]
        #tmpDF = construction_corDF1[(construction_corDF1["J"]==J)&(construction_corDF1["SNR1"]==SNR1)&(construction_corDF1["SNR2"]==SNR2)]
        tmpDF = Phi_align_corDF1[(Phi_align_corDF1["SNR1"] == SNR1) & (Phi_align_corDF1["rank"] == SNR2)]
        g = sns.boxplot(ax = axes[iSNR1,iSNR2],x="rate",
                    y='correlation',
                    hue='method',
                    palette=my_pal,
                    data=tmpDF);
        axes[iSNR1, iSNR2].set_ylim([ymin, ymax])
        if iSNR2 == 0:
            axes[iSNR1, iSNR2].set_ylabel("SNR1="+str(int(SNR1)))
        if iSNR2 != 0:
            #axes[iSNR1,iSNR2].get_yaxis().set_visible(False)
            #axes[iSNR1,iSNR2].get_yaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_ylabel(None)
        if iSNR1 != 2:
            axes[iSNR1,iSNR2].get_xaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_xlabel(None)
        if iSNR1 == 2:
            axes[iSNR1, iSNR2].tick_params(axis='x', rotation=75)
            axes[iSNR1, iSNR2].set_xlabel("Comp"+str(int(SNR2)+1))
        g.legend_.remove()


fig.legend(title=None, labels=sp_name,
            handles=[spaco,   spaco_],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})
fig.tight_layout()
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(hspace=0.2)
fig.show()

FIG_file1 =  curpath + '/results/reconstruction_view_U.png'
fig.savefig(FIG_file1)
plt.close(fig)


method_names_plot = ['SupCP_functional','SupCP_random']
construction_corDF1 = pd.melt(construction_corDF,
        id_vars= ["J", "SNR1", "SNR2", "rate"],
        value_vars= method_names_plot,
                              var_name='method', value_name="correlation")

construction_corDF1=construction_corDF1[~np.isnan(construction_corDF1['correlation'])]
sns.set_context("notebook", font_scale=1.2)
#sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False, sharex= True)

my_pal = {"spaco": "darkred", "spaco_": "coral", "SupCP_functional": "steelblue",
          "SupCP_random":"cornflowerblue", "cp":"darkcyan",
          "Raw":"lightgray"}

spaco = mpatches.Patch(color="darkred")
spaco_ = mpatches.Patch(color="coral")
SupCP_functional = mpatches.Patch(color= "steelblue")
SupCP_random=mpatches.Patch(color= "cornflowerblue")
cp =mpatches.Patch(color= "darkcyan")
Raw=mpatches.Patch(color= "lightgray")
#sp_name = ['SPACO', 'SPACO-', 'SupCP', 'SupCP_rd', "CP", "Raw"]
sp_name = ['SupCP', 'SupCP_random']

for iSNR1 in np.arange(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    #tmpDF1 = construction_corDF1[(construction_corDF1["J"] == J) & (construction_corDF1["SNR1"] == SNR1)]
    tmpDF1 = construction_corDF1[(construction_corDF1["SNR1"] == SNR1)]
    ymax = tmpDF1["correlation"].max()+0.1
    if ymax > 1:
        ymax = 1
    ymin = tmpDF1["correlation"].min()-0.05
    for iSNR2 in np.arange(len(SNR2s)):
        SNR2 = SNR2s[iSNR2]
        #tmpDF = construction_corDF1[(construction_corDF1["J"]==J)&(construction_corDF1["SNR1"]==SNR1)&(construction_corDF1["SNR2"]==SNR2)]
        tmpDF = construction_corDF1[(construction_corDF1["SNR1"] == SNR1) & (construction_corDF1["SNR2"] == SNR2)]
        tmpDF["J_rate"] = ['J'+str(tmpDF["J"].iloc[i])+"_r"+str(str(tmpDF["rate"].iloc[i])) for i in np.arange(tmpDF.shape[0])]
        g = sns.boxplot(ax = axes[iSNR1,iSNR2],x="J_rate",
                    y='correlation',
                    hue='method',
                    palette=my_pal,
                    data=tmpDF);
        axes[iSNR1, iSNR2].set_ylim([ymin, ymax])
        if iSNR2 == 0:
            axes[iSNR1, iSNR2].set_ylabel("SNR1="+str(int(SNR1)))
        if iSNR2 != 0:
            #axes[iSNR1,iSNR2].get_yaxis().set_visible(False)
            axes[iSNR1,iSNR2].get_yaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_ylabel(None)
        if iSNR1 != 2:
            axes[iSNR1,iSNR2].get_xaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_xlabel(None)
        if iSNR1 == 2:
            axes[iSNR1, iSNR2].tick_params(axis='x', rotation=75)
            axes[iSNR1, iSNR2].set_xlabel("SNR2="+str(int(SNR2)))
        g.legend_.remove()


fig.legend(title=None, labels=sp_name,
            handles=[SupCP_functional,  SupCP_random],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})
fig.tight_layout()
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(hspace=0.2)
#fig.show()
FIG_file1 =  curpath + '/results/reconstruction_view_init.png'
fig.savefig(FIG_file1)
plt.close(fig)




method_names_plot = ['spaco', 'spaco_', 'SupCP_functional','cp']
construction_corDF1 = pd.melt(construction_corMissDF,
        id_vars= ["J", "SNR1", "SNR2", "rate"],
        value_vars= method_names_plot,
                              var_name='method', value_name="correlation")

construction_corDF1=construction_corDF1[~np.isnan(construction_corDF1['correlation'])]
sns.set_context("notebook", font_scale=1.2)
#sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False, sharex= True)
A = construction_corDF1[["J","SNR1","SNR2","rate"]]
A.drop_duplicates()

A = construction_corDF1[["J","method"]]
A.drop_duplicates()

my_pal = {"spaco": "darkred", "spaco_": "coral", "SupCP_functional": "steelblue",
          "SupCP_random":"cornflowerblue", "cp":"darkcyan",
          "Raw":"lightgray"}

spaco = mpatches.Patch(color="darkred")
spaco_ = mpatches.Patch(color="coral")
SupCP_functional = mpatches.Patch(color= "steelblue")
SupCP_random=mpatches.Patch(color= "cornflowerblue")
cp =mpatches.Patch(color= "darkcyan")
Raw=mpatches.Patch(color= "lightgray")
#sp_name = ['SPACO', 'SPACO-', 'SupCP', 'SupCP_rd', "CP", "Raw"]
sp_name = ['SPACO', 'SPACO-', 'SupCP', "CP"]

for iSNR1 in np.arange(len(SNR1s)):
    SNR1 = SNR1s[iSNR1]
    #tmpDF1 = construction_corDF1[(construction_corDF1["J"] == J) & (construction_corDF1["SNR1"] == SNR1)]
    tmpDF1 = construction_corDF1[(construction_corDF1["SNR1"] == SNR1)]
    ymax = tmpDF1["correlation"].max()+0.1
    if ymax > 1:
        ymax = 1
    ymin = tmpDF1["correlation"].min()-0.05
    for iSNR2 in np.arange(len(SNR2s)):
        SNR2 = SNR2s[iSNR2]
        #tmpDF = construction_corDF1[(construction_corDF1["J"]==J)&(construction_corDF1["SNR1"]==SNR1)&(construction_corDF1["SNR2"]==SNR2)]
        tmpDF = construction_corDF1[(construction_corDF1["SNR1"] == SNR1) & (construction_corDF1["SNR2"] == SNR2)]
        tmpDF["J_rate"] = ['J'+str(tmpDF["J"].iloc[i])+"_r"+str(str(tmpDF["rate"].iloc[i])) for i in np.arange(tmpDF.shape[0])]
        g = sns.boxplot(ax = axes[iSNR1,iSNR2],x="J_rate",
                    y='correlation',
                    hue='method',
                    palette=my_pal,
                    data=tmpDF);
        axes[iSNR1, iSNR2].set_ylim([ymin, ymax])
        if iSNR2 == 0:
            axes[iSNR1, iSNR2].set_ylabel("SNR1="+str(int(SNR1)))
        if iSNR2 != 0:
            #axes[iSNR1,iSNR2].get_yaxis().set_visible(False)
            axes[iSNR1,iSNR2].get_yaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_ylabel(None)
        if iSNR1 != 2:
            axes[iSNR1,iSNR2].get_xaxis().set_ticklabels([])
            axes[iSNR1, iSNR2].set_xlabel(None)
        if iSNR1 == 2:
            axes[iSNR1, iSNR2].tick_params(axis='x', rotation=75)
            axes[iSNR1, iSNR2].set_xlabel("SNR2="+str(int(SNR2)))
        g.legend_.remove()


fig.legend(title=None, labels=sp_name,
            handles=[spaco, spaco_, SupCP_functional, cp],
            fancybox=True, shadow=True, ncol=len(sp_name), loc='upper center',
           prop={'size': 12})
fig.tight_layout()
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(hspace=0.2)
#fig.show()
FIG_file1 =  curpath + '/results/reconstruction_view_signalmat_miss.png'
fig.savefig(FIG_file1)
plt.close(fig)
