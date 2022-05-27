import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../catchall')
import datahelp as dm 

def autocorr(x, lags):
    '''fft-based method'''
    if len(x.shape)==1:
        x = x.reshape(1,-1) # timeseries must be along the row-axis ...!!!
    tlen = np.shape(x)[1]
    x = x - np.mean(x,1).reshape(-1,1) # subtract mean
    
    # fsize = 2**np.ceil(np.log2(2*n-1)).astype('int') # next power of 2
    # xf = np.fft.fft(x, n=fsize, axis=1)
    xf = np.fft.fft(x, axis=1) # fft 
    sxx = xf.conjugate()*xf/tlen # compute power
    corr = np.fft.ifft(sxx, axis=1).real
    
    var = np.var(x,1).reshape(-1,1)
    corr = corr/var # normalize, shape=[1,tlen]
    corr = corr[:,:len(lags)]
    return corr.sqeeuze()
#%% Power spectrum and PSD
def powerspectrum_nowindow(x, fs): # time-axis along 0, same as Matlab
    tlen = x.shape[0]
    y = np.fft.fft(x, axis=0)
    ps = 2*np.abs(y*y.conjugate())/tlen # why 2? Because you're going to use the half of fft output, by multiplying 2 --> sum(x**2)=sum(ps)
    f = np.linspace(0, fs/2, int(tlen/2 + 1) )
    ps = ps[:len(f)]
    return ps, f

def powerspectrum_window(x, fs, segsize, segmove, window='hamming', scaling='density'): # time-axis along 0, same as Matlab
    '''This function generates the equivalent output as scipy.signal.welch'''
    # segsize=200;segmove=100;fs=200
    x_segdata = dm.windowingSlides(x.T, segsize, segmove) # [segnum,ch,segsize]
    
    ### Multiply window function ###
    if window=='boxcar' or window==None:
        x_window = np.ones(segsize)
    else:
        exec('x_window = np.' + window + '(segsize)')
    x_segdata = x_segdata*x_window
    
    ### FFT ###
    y_segdata = np.fft.fft(x_segdata, axis=-1) # [segnum,ch,segsize]
    
    ### Compute Power ###
    S = 2*np.abs(y_segdata*y_segdata.conjugate()) # why 2? Because you're going to use the half of fft output, by multiplying 2 --> sum(x**2)=sum(ps)
    
    ### Scale ###
    if scaling=='density':
        scale_factor = np.expand_dims(S.sum(-1), -1) # sum will be appx. 1.
    elif scaling=='spectrum':
        scale_factor = (x_window.sum())**2
        
    S = S/scale_factor
    pxx = np.mean(S,0)
    
    ### Frequency ###
    f = np.linspace(0, fs/2, int(segsize/2+1))
    pxx = pxx[:len(f)]
    return pxx, f
'''
Comparison with scipy.signal.welch 
    https://stackoverflow.com/questions/57828899/prefactors-computing-psd-of-a-signal-with-numpy-fft-vs-scipy-signal-welch/57877348#57877348
    import scipy.signal
    x = np.random.randn(5000,)
    f, pxx = scipy.signal.welch(x, fs=200, window='boxcar', nperseg=200, noverlap=100, scaling='density')
    plt.figure()
    plt.plot(f, pxx)
    
    pxx2,f = powerspectrum_window(x, 200, 200, 100, scaling='density', window='boxcar')
    plt.plot(f, pxx2)
'''
#%% Coherence
import scipy.signal
def coherence(x, fs, segsize, segmove, window='hamming'):
    x_segdata = dm.windowingSlides(x.T, segsize, segmove) # [segnum,ch,segsize]
    
    ### Multiply window function ###
    x_window = scipy.signal.windows.get_window(window, segsize)
    x_segdata = x_segdata*x_window
    
    ### FFT ###
    y_segdata = np.fft.fft(x_segdata, axis=-1)
    f = np.linspace(0, fs/2, int(segsize/2+1))

    segnum, ch, _ = y_segdata.shape
    coh = []
    for c in range(ch-1): # c=0
        y1 = y_segdata[:,c].reshape(segnum,1,segsize)#[segnum,1,nfft]
        y2 = np.conj(y_segdata[:,c+1:])#[segnum,ch?,nfft]
        
        sij = np.mean(y1*y2, 0) # [ch?, nfft]
        sii = np.mean(np.abs(y1*y1.conjugate()), 0) # [ch?,nfft]
        sjj = np.mean(np.abs(y2*y2.conjugate()), 0) # [ch?,nfft]
        coh.append(np.abs(sij)/np.sqrt(sii*sjj))
    
    coh = np.row_stack(coh)[:,:len(f)]
    return coh.squeeze(), f
#%% Filters
def zerofilt_fir(x, bp, fs, ntaps, ftype, axis=-1):
    '''pass_zero{True, False, ‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’}, optional
    If True, the gain at the frequency 0 (i.e. the “DC gain”) is 1. If False, the DC gain is 0. Can also be a string argument for the desired filter type (equivalent to btype in IIR design functions).
    '''
    taps = scipy.signal.firwin(ntaps, bp, nyq=fs/2, pass_zero=ftype, window='hamming', scale=False)
    xFilt = scipy.signal.filtfilt(taps, 1, x, axis=axis)
    return xFilt

def zerofilt_bttw(x, bp, fs, f_order=4, ftype = 'band', axis = -1):
    # ftype: 'band', or 'bandstop'
    nyq = fs/2
    # b, a = scipy.signal.butter(f_order,bp/nyq, btype=ftype)
    # xFilt = scipy.signal.filtfilt(b, a, x, axis = axis)
    sos = scipy.signal.butter(f_order, bp/nyq, btype=ftype, output='sos') #!!! what is sos???
    xFilt = scipy.signal.sosfiltfilt(sos, x, axis = axis)
    return xFilt
    
def notch_iir(x, f0s, fs, axis = -1, Q =30.0):
    # Q: Quality factor. Dimensionless parameter that characterizes 
    # notch filter -3dB bandwidth bw relative to its center frequency, Q=w0/bw.
    for f0 in f0s:
        b, a = scipy.signal.iirnotch(f0, Q, fs)
        x = scipy.signal.filtfilt(b, a, x, axis = axis)
    return x
'''

b = scipy.signal.firwin(80, [10,20], fs=200, window='hamming',pass_zero='bandpass')
w, h = scipy.signal.freqz(b)
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.xlabel('Frequency [rad/sample]')
'''













