#!/usr/bin/env python

"""extract.py: Extract pumping-like traces from kymographs etc.."""
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.signal import find_peaks
from skimage.util import view_as_windows
import matplotlib.pylab as plt
import pharaglow.features as pg
from pharaglow import util

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
    

def extractMaxWidths(df, cut = 60):
    """takes dataframe with one particle and returns width kymograph and distance between peaks."""
    w = np.array([pg.scalarWidth(row)[:,0] for row in df['Widths']])
    return w, np.argmax(w[:,5:cut], axis = 1)+5, np.argmax(w[:,cut:-5], axis = 1)+cut


def extractImages(df, key):
    """get images from dataframe."""
    return [np.array(im) for im in df[key].values]


def minMaxDiffKymo(df, key):
    """get the min and max intensity in a kymograph."""
    kymo = np.array(df[key].values)
    kymo = np.array([np.interp(np.linspace(0, len(row), 100), np.arange(len(row)), np.array(row)) \
                      for row in kymo]).T
    dkymo = np.diff(kymo[0:], axis = 0)
    return np.max(dkymo, axis = 0), -np.min(dkymo, axis = 0)


def pumpingMetrics(traj, params):
    """given a dataframe with one trajectory, extract many pumping metrics."""
    df = pd.DataFrame()

    _, xl, xu = extractMaxWidths(traj, params['cut'])
    # difference of widths
    pwidth = xu -xl
    # get trajectory wiggles
    dv = np.diff(traj['xw']-traj['x'])**2+np.diff(traj['yw']-traj['y'])**2
    dv = np.pad(dv, [1,0], mode = 'constant')
    # normal kymograph
    pkymo = extractKymo(traj, key = 'Kymo')
    # weighted normal kymograph
    pkymoW = extractKymo(traj, key = 'WeightedKymo')
    # normal kymograph gradient
    pkymoGrad = extractKymo(traj, key = 'KymoGrad')
    # normal kymograph gradient weighted
    pkymoGradW = extractKymo(traj, key = 'WeightedKymoGrad')
    # measure pumps by min/max in kymograph
    maxpump, minpump = minMaxDiffKymo(traj, key = 'Kymo')
    # measure pumps by skew of difference intensity
    imgs = extractImages(traj, 'Straightened')
    pwarp = [np.abs(skew(im[0:], axis = None)) for im in np.diff(imgs, axis =0)]
    pwarp = np.pad(pwarp, [1,0], mode = 'constant')
    pwarpmean = [np.mean(np.abs(im[0:])) for im in np.diff(imgs, axis =0)]
    pwarpmean = np.pad(pwarpmean, [1,0], mode = 'constant')
    pwarpmax = [np.max(np.abs(im[0:])) for im in np.diff(imgs, axis =0)]
    pwarpmax = np.pad(pwarpmax, [1,0], mode = 'constant')
    
    df = df.append([{'Bulb Distance': pwidth,
                             'CMS': dv,
                             'Kymo':pkymo,
                             'WeightedKymo':pkymoW,
                             'KymoGrad': pkymoGrad,
                             'WeightedKymoGrad': pkymoGradW,
                             'maxPump': maxpump,
                             'minPump': minpump,
                             'pwarp': pwarp,
                             'meanDiff': pwarpmean,
                             'maxDiff': pwarpmax
                             },])
    return df



def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    t0: how many sigma away to call it an outlier
    '''
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
    """preprocess a trace with rolling window brackground subtraction."""
    bg = p.rolling(w_bg, min_periods=1, center=True, win_type=win_type_bg).median()
    return (p - bg).rolling(w_sm, min_periods=1, center=True, win_type=win_type_sm).mean(), bg


def find_pumps(p, heights = np.arange(0.01, 5, 0.1), min_distance = 5, sensitivity = 0.99, **kwargs):
    """peak detection in a background subtracted trace assuming real 
        peaks have to be at least min_distance samples apart."""
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
    

def pumps(data):
    straightIms = np.array([im for im in data['Straightened'].values])
    print(straightIms.shape)
    k = np.max(np.std(straightIms, axis =2), axis =1)#-np.mean(straightIms, axis =2)
    #k = -np.max(np.median(straightIms, axis =2), axis =1)
    #k = np.min(np.mean(straightIms[:,150:,], axis =2), axis =1)
    return np.ravel(k)



