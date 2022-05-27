import numpy as np

def delayRecons(data, m, L):
    '''
    data must be along the "columna" axis (0), same as your Matlab code.
    the second dimension has the "delayed vector"
    Ddata[t,:,c] : 
            time of Ddata[-t,0,c] < time of Ddata[-t,1,c]
            i.e., later element is "future".
    '''
    if data.ndim == 1:
        data = data.reshape(-1,1)
    data_len, ch = np.shape(data)
    Ddata = np.zeros((data_len - L*(m-1), m, ch), dtype=data.dtype)
    for j in range(m):# j+=1
        i1 = j*L
        i2 = data_len- (m-(j+1))*L
        Ddata[:,j,:] = data[i1:i2,:]
    return Ddata

def delayRecons_1d(data, m, L):
    # data must be along the "columna" axis (0), same as Matlab
    data = data.reshape(-1,1)
    data_len = np.shape(data)[0]
    Ddata = np.zeros((data_len - L*(m-1), m), dtype=data.dtype)
    for j in range(m):# j+=1
        i1 = j*L
        i2 = data_len- (m-(j+1))*L
        Ddata[:,j] = data[i1:i2,0]
    return Ddata

def delayReconsTrials(data, m, L):
    data_len, ch, n_trials = np.shape(data)
    Ddata = np.zeros((data_len - L*(m-1), m, ch, n_trials))
    for j in range(m):# j+=1
        i1 = j*L
        i2 = data_len- (m-(j+1))*L
        Ddata[:,j,:,:] = data[i1:i2,:,:]
    return Ddata

def correlation_sum(Ddata, r, n_min, metric='euclidean'):
    '''
    correlation sum = (number of small pairwise distances)/(number of all pairwise distances)
        how poitns are densily distributed in a space?
        See your OneNote for interpretation and how it is related to Shannon entropy & information content.
    '''
    from scipy.spatial.distance import pdist, squareform
    D = pdist(Ddata, metric) # [N,d] --> [N*(N-1)/2,]
    D_close = D<r
    D_close = squareform(D_close) # [N,N]
    N = np.shape(D_close)[0]
    
    for i in range(n_min):
        np.fill_diagonal(D_close[i+1:], 0) # too much close points in time isn't valid
        np.fill_diagonal(D_close[:,i+1:], 0) # too much close points in time isn't valid
    C = np.nansum(D_close)
    
    # normalization
    n_min = n_min-1
    denom = (N-n_min)*(N-n_min-1)
    C = C/denom
    return C




















