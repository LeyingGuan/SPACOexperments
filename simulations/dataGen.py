import numpy as np
import sys
import random



def dataGen(I, T, J, q, rate, s=3, K0 = 3, SNR1 = 1.0, SNR2 = 3.0):
    Phi0 = np.zeros((T, K0))
    Phi0[:,0] = 1.0
    Phi0[:,1] = np.arange(T)/T
    Phi0[:, 1] = np.sqrt(1-Phi0[:,1]**2)
    Phi0[:,2] = (np.cos((np.arange(T))/T * 4*np.pi))
    for k in np.arange(K0):
        Phi0[:, k] = Phi0[:, k]
        Phi0[:, k] = Phi0[:, k] /(np.sqrt(np.mean(Phi0[:, k] ** 2))) * (np.log(J)+np.log(T))/np.sqrt(I*T*rate) *SNR1
    V0 = np.random.normal(size=(J, K0))*1.0/np.sqrt(J)
    Z = np.random.normal(size =(I, q))
    U = np.random.normal(size =(I,K0))
    beta = np.zeros((q,K0))
    for k in np.arange(K0):
        if q > 0:
            if s > q:
                s = q
            beta[:s,k] = np.random.normal(size = (s)) * np.sqrt(np.log(q)/I) * SNR2
            U[:,k] = U[:,k]+np.matmul(Z, beta[:,k])
        U[:,k] = U[:,k] - np.mean(U[:,k])
        U[:,k] = U[:,k]/np.std(U[:,k])
    Xcomplete = np.random.normal(size=(I, T, J)) * 1.0
    T0 = np.arange(T)
    signal_complete = np.zeros(Xcomplete.shape)
    PhiV0 = np.zeros((T, J, K0))
    for k in np.arange(K0):
        PhiV0[:,:,k] = np.matmul(Phi0[:,k].reshape((T,1)), V0[:,k].reshape(1,J))
    for i in np.arange(I):
        for k in np.arange(K0):
            signal_complete[i, :, :] += PhiV0[:,:,k] * U[i,k]
        Xcomplete[i, :, :] += signal_complete[i, :, :]
    Obs = np.ones(Xcomplete.shape, dtype=int)
    Xobs = Xcomplete.copy()
    for i in np.arange(I):
        ll = T0
        tmp = np.random.choice(T0,replace=False,size=T - int(rate * T))
        Obs[i, ll[tmp], :] = 0
        Xobs[i, ll[tmp], :] = np.nan
    return Xcomplete, signal_complete , Xobs, Obs, T0, Phi0, V0, U, PhiV0, Z, beta

Is = [100]
Js = [10, 500]
T = 30
qs = [100]
rates = [1.0, 0.5, 0.1]
SNR1s = [1.0,  5.0]
SNR2s = [0.0, 1.0, 10.0]
iteration = 10

np.random.seed(1)
seeds = np.zeros(100, dtype = int)
for i in np.arange(100):
    seeds[i] = random.randrange(0, 100000)

for it in np.arange(iteration):
    np.random.seed(seeds[it])
    for i1 in np.arange(len(Is)):
        for i2 in np.arange(len(Js)):
            for i3 in np.arange(len(qs)):
                for i4 in np.arange(len(rates)):
                    for i5 in np.arange(len(SNR1s)):
                        for i6 in np.arange(len(SNR2s)):
                            I = Is[i1]
                            J = Js[i2]
                            q = qs[i3]
                            rate = rates[i4]
                            SNR1 = SNR1s[i5]
                            SNR2 = SNR2s[i6]
                            Xcomplete, signal_complete , Xobs, Obs, T0, Phi0, V0, U, PhiV0, Z, beta0 = \
                                dataGen(I, T, J, q, rate, s=3, K0 = 3, SNR1 = SNR1, SNR2 = SNR2)
                            file = 'data/data_subj'+str(I)+'_time'+str(T)+'_feature'+str(J)+\
                                     '_Z'+str(q)+'_rate'+str(rate)+'_SNR1'+str(SNR1)+'_SNR2'+\
                                     str(SNR2)+'_iteration'+str(it)+'.npz'
                            np.savez(file, Xcomplete = Xcomplete,
                                     signal_complete = signal_complete, Xobs = Xobs, Obs = Obs, T0 = T0,
                                     Phi0 = Phi0, V0 = V0, U = U, PhiV0 = PhiV0, Z = Z, beta0 = beta0)





