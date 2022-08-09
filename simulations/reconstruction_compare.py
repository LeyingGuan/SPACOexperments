import spaco
import numpy as np
import sys
import os
import pickle

# I = 100; T = 30; q = 100;J = 10;SNR1 = 1.0;SNR2 = 5.0;rate = 0.1
# curpath = '/home/lg689/project/SPACOresults'
# it = 0

I = int(sys.argv[1])
T = int(sys.argv[2])
J = int(sys.argv[3])
q = int(sys.argv[4])
rate = float(sys.argv[5])
SNR1 = float(sys.argv[6])
SNR2 = float(sys.argv[7])
iteration = 20
curpath = "/home/lg689/project/SPACOresults"

comparison_res_list = []
res_file_name = curpath + '/results/construction/data_subj' + str(
    I) + '_time' + str(T) + '_feature' + str(J) + \
                '_Z' + str(q) + '_rate' + str(
    rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
                str(SNR2) + '.pickle'

for it in np.arange(iteration):
    print("##############"+str(it)+"####################")
    file = curpath+'/data/data_subj' + str(I) + '_time' + str(T) + '_feature' + str(J) + \
           '_Z' + str(q) + '_rate' + str(rate) + '_SNR1' + str(SNR1) + '_SNR2' + \
           str(SNR2) + '_iteration' + str(it) + '.npz'
    data = np.load(file)
    rank = 3; max_iter = 30
    comparison_res_list.append(spaco.comparison_pipe(Phi0=data['Phi0'],
                                     V0 =data['V0'],
                                     U0 = data['U'],
                                     X = data['Xobs'],
                                     Z =  data['Z'],
                                     O = data['Obs'],
                                     time_stamps = data['T0']))

    comparison_res_list[it].compare_run(rank = rank, max_iter = max_iter)

    for method in comparison_res_list[it].eval_dict.keys():
        if method != "empirical":
            print(method)
            print(comparison_res_list[it].eval_dict[method].alignment)

    print('empirical')
    print(comparison_res_list[it].eval_dict['empirical'])
    comparison_res_list[it].X = None
    comparison_res_list[it].O = None
    comparison_res_list[it].signal = None
    comparison_res_list[it].spaco_fit.intermediants = None
    comparison_res_list[it].spaco_fit.O = None
    comparison_res_list[it].spaco_fit.X = None
    comparison_res_list[it].spaco_fit.X0 = None

with open(res_file_name, 'wb') as handle:
    pickle.dump(comparison_res_list, handle,
                protocol=pickle.HIGHEST_PROTOCOL)


# with open(res_file_name, 'rb') as handle:
#     b = pickle.load(handle)






