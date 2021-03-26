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
    """extract the difference of the kymo."""
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



def nanOutliers(data, window_size, n_sigmas=3):
    """using a Hampel filter to detect outliers and replace them with nans. 
    The algorithm assuems the underlying series should be gaussian. this could be changed by changing the k parameter.
    Data can contain nans, these will be ignored for median calculation.
    data: (M, N) array of M timeseries with N samples each. 
    windowsize: integer. This is half of the typical implementation.
    n_sigmas: float or integer
    """
    # deal with N = 1 timeseries to conform to (M=1,N) shape
    if len(data.shape)<2:
        data = np.reshape(data, (1,-1))
    k = 1.4826 # scale factor for Gaussian distribution
    M, N = data.shape # store data shape for quick use below
    # pad array to achieve at least the same size as original
    paddata = np.pad(data.copy(), pad_width= [(0,0),(window_size//2,window_size//2)], \
                     constant_values=(np.median(data)), mode = 'constant')
    # we use strides to create rolling windows. Not nice in memory, but good in performance.
    # because its meant for images, it creates some empty axes of size 1.
    tmpdata = view_as_windows(paddata, (1,window_size))
    # crop data to center 
    tmpdata = tmpdata[:M, :N]
    x0N = np.nanmedian(tmpdata, axis = (-1, -2))
    s0N =k * np.nanmedian(np.abs(tmpdata - x0N[:,:,np.newaxis, np.newaxis]), axis = (-1,-2))
    # hampel condition
    hampel = (np.abs(data-x0N) - (n_sigmas*s0N))
    indices = np.where(hampel>0)
    #cast to float to allow nans in output data
    newData = np.array(data, dtype=float)
    newData[indices] = x0N[indices]
    return newData, indices

def pumps(data):
    straightIms = np.array([im for im in data['Straightened'].values])
    print(straightIms.shape)
    k = np.max(np.std(straightIms, axis =2), axis =1)#-np.mean(straightIms, axis =2)
    #k = -np.max(np.median(straightIms, axis =2), axis =1)
    #k = np.min(np.mean(straightIms[:,150:,], axis =2), axis =1)
    return np.ravel(k)

def manAutoComp(time0, pump0, v, manp, inside):
    # crop automatic and manual to the same time frame
    mi, ma = int(np.nanmax([manp.min(), np.nanmin(time0)]))-5, int(np.nanmin([manp.max(), np.nanmax(time0)]))
    #print(mi,ma)
    manp = manp[(manp>=mi)&(manp<=ma)]
    time = time0[(time0>=mi)&(time0<=ma)]
    pump = pump0[(time0>=mi)&(time0<=ma)]
    v = v[(time0>=mi)&(time0<=ma)]
    inside = inside[(time0>=mi)&(time0<=ma)]
    # extract outliers and remove trends
    #pump = pump- util.smooth(pump, 600)
    
    pump, ind = nanOutliers(pump.values, window_size = 300, n_sigmas=3)
    pump, ind = pump[0], ind[1]
    pump = pump - util.smooth(pump, 30)
    pump = pd.Series(pump)
    
    # rescale to percentile
    pump -= np.mean(pump)
    pump/= np.percentile(pump, [0.5])#/2#/5
    return time, pump, manp, v, inside

## ROC curve for peak detection
def rocPeaks(pump, pars):
    ps, roc = [], []
    w = 2
    for p in pars:
        peaks = find_peaks(pump, height=(p, 1.75), threshold=None, distance=5, prominence=None,\
                    width=None, wlen=10, rel_height=None, plateau_size=None)[0]
        meanPeak = np.array([pump[peak-w:peak+w+1] for peak in peaks if (peak +w <len(pump)) and peak-w>0]).T
        ps.append(peaks)
        roc.append(meanPeak)
    return ps, roc
    

def preprocess(p, w_bg, w_sm, **kwargs):
    bg = p.rolling(w_bg, min_periods=1, center=True, win_type='hamming').mean()
    return (p - bg).rolling(w_sm, min_periods=1, center=True, win_type='parzen').mean(), bg


def find_pumps(p, heights = np.arange(0.01, 5, 0.1), min_distance = 5, sensitivity = 0.99, **kwargs):
    """peak detection in a background subtracted trace assuming real 
        peaks have to be at least min_distance samples apart."""
    tmp = []
    all_peaks = []
    # find peaks at different heights
    for h in heights:
        peaks = find_peaks(p, height = h,threshold =0.0)[0]
        tmp.append([len(peaks), np.mean(np.diff(peaks)>=min_distance)])
        all_peaks.append(peaks)
    tmp = np.array(tmp)
    # set the valid peaks score to zero if no peaks are present
    tmp[:,1][~np.isfinite(tmp[:,1])]= 0
    # calculate random distribution of peaks in a series of length l (actually we know the intervals will be exponential)
    null = []
    l = len(p)
    for npeaks in tmp[:,0]:
        locs = np.random.randint(0,l,(500, int(npeaks)))
        # calculate the random error rate - and its stdev
        null.append([np.mean(np.diff(np.sort(locs), axis =1)>=min_distance), np.std(np.mean(np.diff(np.sort(locs), axis =1)>=5, axis =1))])
    null = np.array(null)
    # now find the best peak level - larger than random, with high accuracy
    # subtract random level plus 1 std:
    metric_random = tmp[:,1] - (null[:,0]+null[:,1])
    # check where this is still positive and where the valid intervals are 1 or some large value
    valid = np.where((metric_random>0)*(tmp[:,1]>=sensitivity))[0]
    if len(valid)>0:
        peaks = all_peaks[valid[np.argmax(tmp[:,0][valid])]]
    else:
        return [], tmp, null
    return peaks, tmp, null


def bestMatchPeaks(pump, wsDetrend = 10 , wsOutlier = 2, wsDetrendLocal = 2, prs = np.linspace(0.1,5,50)):
    """DEPRECATED: now calls new find_pumps function."""
    ### define best match
    pump = preprocess(pump, wsDetrend, wsDetrendLocal)
    return find_pumps(pump, heights = prs, min_distance = 5, sensitivity = 0.95)
