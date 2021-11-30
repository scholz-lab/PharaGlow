#!/usr/bin/env python

"""extract.py: Extract pumping-like traces from kymographs etc.."""
import numpy as np
from scipy.signal import find_peaks


def alignKymos(ar):
    """Align a kymograph by largest correlation.

    Args:
        ar (numpy,array): 2d array with each row a kymograph line

    Returns:
        numpy.array: array with each row shifted for optimal correlation with the first row.
    """    
    sample = ar[0]
    sample =sample - np.mean(sample)
    ar2 = np.zeros(ar.shape)
    #shifts = np.zeros(len(ar))
    for ri, row in enumerate(ar):
        row =row - np.mean(row)
        row =row/np.std(row)
        corr = np.correlate(sample,row,mode='full')
        shift = int(np.argmax(corr))
        ar2[ri] = np.roll(ar[ri], shift)
    return ar2


def extractKymo(df, key):
    """extract the difference of the kymo.

    Args:
        df (pandas.DataFrame): a pharaglow results dataframe
        key (str): a column name in dataframe e.g. 'kymo'

    Returns:
        numpy.array: the diff between adjacent kymograph rows
    """    
    
    # need to get rid of none values and such
    kymo = [np.array(list(filter(None.__ne__,row))) for row in df[key].values]
    kymo = np.array([np.interp(np.linspace(0, len(row), 100), np.arange(len(row)), np.array(row)) \
                      for row in kymo])
    kymo = alignKymos(kymo).T
    return np.nansum(np.abs(np.diff(kymo[0:], axis = 0)), axis = 0)
    

def hampel(vals_orig, k=7, t0=3):
    """Implements a Hampel filter (code from E. Osorio, Stackoverflow).

    Args:
        vals_orig (list, numpy.array): series of values to filter
        k (int, optional): window size to each side of the sample eg. 7 is 3 left and 3 right of the sample value. Defaults to 7.
        t0 (int or float, optional): how many sigma away is an outlier. Defaults to 3.
    """    
    #Make copy so original not edited
    vals = vals_orig.copy()
    
    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True, min_periods = 1).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True, min_periods=1).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)


def preprocess(p, w_bg, w_sm, win_type_bg = 'hamming', win_type_sm = 'boxcar', **kwargs):
    """preprocess a trace with rolling window brackground subtraction.

    Args:
        p (numpy.array): input signal or time series
        w_bg (int): background window size
        w_sm (int): smoothing window size
        win_type_bg (str, optional): background window type. Defaults to 'hamming'.
        win_type_sm (str, optional): smoothing window type. Defaults to 'boxcar'.

    Returns:
        numpy.array: background subtracted and smoothed signal
    """    
    
    bg = p.rolling(w_bg, min_periods=1, center=True, win_type=win_type_bg).median()
    return (p - bg).rolling(w_sm, min_periods=1, center=True, win_type=win_type_sm).mean(), bg


def find_pumps(p, heights = np.arange(0.01, 5, 0.1), min_distance = 5, sensitivity = 0.99, **kwargs):
    """ peak detection method finding the maximum number of peaks while allowing fewer than sensitivity fraction of peaks that are closer than a min_distance.

    Args:
        p (numpy,array): input signal with peaks
        heights (list, optional): peak prominence values to test. Defaults to np.arange(0.01, 5, 0.1).
        min_distance (int, optional): distance peaks should be apart. Defaults to 5.
        sensitivity (float, optional): how many peaks can violate min_distance. Defaults to 0.99.

    Returns:
        list: peak indices
        numpy.array: peak number and fraction of valid peaks for all heights
        numpy.array: mean and standard deviation fraction of valid peaks for all heights in a random occurance

    """    
   
    tmp = []
    all_peaks = []
    # find peaks at different heights
    for h in heights:
        peaks = find_peaks(p, prominence = h, **kwargs)[0]
        tmp.append([len(peaks), np.mean(np.diff(peaks)>=min_distance)])
        all_peaks.append(peaks)
    tmp = np.array(tmp)
    # set the valid peaks score to zero if no peaks are present
    tmp[:,1][~np.isfinite(tmp[:,1])]= 0
    # calculate random distribution of peaks in a series of length l (actually we know the intervals will be exponential)
    null = []
    l = len(p)
    for npeaks in tmp[:,0]:
        locs = np.random.randint(0,l,(100, int(npeaks)))
        # calculate the random error rate - and its stdev
        null.append([np.mean(np.diff(np.sort(locs), axis =1)>=min_distance), np.std(np.mean(np.diff(np.sort(locs), axis =1)>=min_distance, axis =1))])
    null = np.array(null)
    # now find the best peak level - larger than random, with high accuracy
    # subtract random level plus 1 std:
    metric_random = tmp[:,1] - (null[:,0]+null[:,1])
    # check where this is still positive and where the valid intervals are 1 or some large value
    valid = np.where((metric_random>0)*(tmp[:,1]>=sensitivity))[0]
    if len(valid)>0:
        #peaks = all_peaks[valid[np.argmax(tmp[:,0][valid])]]
        h = heights[valid[np.argmax(tmp[:,0][valid])]]
        peaks = find_peaks(p, prominence = h, distance = min_distance, **kwargs)[0]
    else:
        return [], tmp, null
    return peaks, tmp, null
    

def pumps(data, key = 'Straightened'):
    """Pumping metric based on the dorsal-vetral signal variation.

    Args:
        data (pandas.DataFrame): dataframe with a column containing straightened images of pharyxes

    Returns:
        numpy.array: signal for all images
    """
    straightIms = np.array([im for im in data[key].values])
    k = np.max(np.std(straightIms, axis =2), axis =1)#-np.mean(straightIms, axis =2)
    return np.ravel(k)
