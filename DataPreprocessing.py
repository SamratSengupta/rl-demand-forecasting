# -*- coding: utf-8 -*-


import numpy as np


def build_state_action(sequence,n,m):
    '''
    Args:
        sequence: Time series data
        n: The number of historical data denoting the current state
        m: The number of prediction steps in advance
    Return:
        state_mat: A matrix contains all states at each time step
        best_action: The optimal action based on each state
    '''
    n_rows = len(sequence)-n-m+1
    state_mat = np.zeros((n_rows,n))
    opt_action = np.zeros(n_rows)
    for i in range(n_rows):
        state_mat[i] = sequence[i:(i+n)]
        opt_action[i] = sequence[i+n+m-1]
    return state_mat,opt_action



def normalization(traindata,testdata):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(traindata)
    traindata_scaled = scaler.transform(traindata)
    testdata_scaled = scaler.transform(testdata)
    
    return traindata_scaled,testdata_scaled
