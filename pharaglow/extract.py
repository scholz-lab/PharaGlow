#!/usr/bin/env python

"""extract.py: Extract pumping-like traces from kymographs etc.."""
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from pyampd.ampd import find_peaks_adaptive
import pandas as pd
from scipy.stats import circmean, circstd


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
    return vals


def preprocess(p, w_bg, w_sm, win_type_bg = 'hamming', win_type_sm = 'boxcar'):
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

    bg = p.rolling(w_bg, min_periods=1, center=True, win_type=win_type_bg).mean()
    return (p - bg).rolling(w_sm, min_periods=1, center=True, win_type=win_type_sm).mean(), bg


def find_pumps(metric, heights = np.arange(0.01, 5, 0.1), min_distance = 5, sensitivity = 0.99, **kwargs):
    """ peak detection method finding the maximum number of peaks while allowing fewer than
    a sensitivity fraction of peaks that are closer than a min_distance.

    Args:
        metric (numpy.array): input signal with peaks
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
        peaks = find_peaks(metric, prominence = h, **kwargs)[0]
        tmp.append([len(peaks), np.mean(np.diff(peaks)>=min_distance)])
        all_peaks.append(peaks)
    tmp = np.array(tmp)
    # set the valid peaks score to zero if no peaks are present
    tmp[:,1][~np.isfinite(tmp[:,1])]= 0
    # calculate random distribution of peaks in a series of length l (actually we know the intervals will be exponential)
    null = []
    l = len(metric)
    for npeaks in tmp[:,0]:
        locs = np.random.randint(0,l,(100, int(npeaks)))
        # calculate the random error rate - and its stdev
        null.append([np.mean(np.diff(np.sort(locs), axis =1)>=min_distance), \
            np.std(np.mean(np.diff(np.sort(locs), axis =1)>=min_distance, axis =1))])
    null = np.array(null)
    # now find the best peak level - larger than random, with high accuracy
    # subtract random level plus 1 std:
    metric_random = tmp[:,1] - (null[:,0]+null[:,1])
    # check where this is still positive and where the valid intervals are 1 or some large value
    valid = np.where((metric_random>0)*(tmp[:,1]>=sensitivity))[0]
    if len(valid)>0:
        #peaks = all_peaks[valid[np.argmax(tmp[:,0][valid])]]
        h = heights[valid[np.argmax(tmp[:,0][valid])]]
        peaks = find_peaks(metric, prominence = h, distance = min_distance, **kwargs)[0]
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
    
    
def preprocess_signal(df, key, w_outlier, w_bg, w_smooth, **kwargs):
    "Use outlier removal, background subtraction and filtering to clean a signal."
    # remove outliers
    sigma = kwargs.pop('sigma', 3)
    # make a copy of the signal
    df.loc[:,f'{key}_clean'] = df[key]
    if w_outlier is not None:
        df[f'{key}_clean'] = hampel(df[f'{key}_clean'], w_outlier, sigma)
    df[f'{key}_clean'],_ = preprocess(df[f'{key}_clean'], w_bg, w_smooth)
    return df


def _illegal_intervals(signal, peaks, min_dist):
    """calculate the fraction of illegal intervals given a prominence cutoff."""
    prom,_,_  = peak_prominences(signal, peaks)
    frac_ill = np.zeros(len(prom))
    for i, p in enumerate(np.sort(prom)):
        tmp_peaks = peaks[prom>p]
        frac_ill[i] = np.sum(np.diff(tmp_peaks)<min_dist)/len(peaks)
    return prom, frac_ill



def _select_valid_peaks(peaks, prom, frac_illegal, sensitivity):
    idx = np.where(frac_illegal <= 1-sensitivity)[0]
    if len(idx) >0:
        min_prom = np.sort(prom)[idx[0]]
        return peaks[prom>min_prom]
    else:
        return []

def _pyampd(signal, adaptive_window, min_distance = None, min_prominence = None, wlen = None):
    peaks = find_peaks_adaptive(signal, window=adaptive_window)
    # remove violating peaks by height or distance
    if min_prominence is not None:
        prom,_,_  = peak_prominences(signal, peaks, wlen)
        peaks = peaks[prom>min_prominence]

    if min_distance is not None:
        rejects = []
        prom,_,_ = peak_prominences(signal, peaks, wlen)
        # calculate peaks with violating intervals
        locs = np.where(np.diff(peaks)<min_distance)[0]
        #print(f'{len(locs)} violating peaks')
        # get the peaks around the offending interval
        for loc in locs:
            local_start = np.max([0, loc-3])
            local_end = np.min([loc+4, len(peaks)])
            local_peaks = peaks[local_start:local_end]
            local_prom = prom[local_start:local_end]
            local_diff = np.where(np.diff(local_peaks)<min_distance)[0]
            #plt.plot(signal[local_peaks[0]: local_peaks[-1]])
            #plt.plot(local_peaks-peaks[0], signal[local_peaks], 'ro')
            while len(local_diff  > 0):
                # remove smallest peak
                min_peak = local_peaks[np.argmin(local_prom)]
                rejects.append(min_peak)
                local_peaks = local_peaks[local_peaks != min_peak]
                local_prom = np.delete(local_prom, local_prom.argmin())
                # check if there are still offending intervals
                local_diff = np.where(np.diff(local_peaks)<min_distance)[0]

        rejects = np.unique(rejects)
        peaks = peaks[~np.isin(peaks, rejects)]
        locs = np.where(np.diff(peaks)<min_distance)[0]
    return peaks
    

def detect_peaks(signal, adaptive_window, min_distance = 4, min_prominence = None, sensitivity=0.95, use_pyampd = True, **kwargs):
    #peaks = find_peaks(signal,scale=adaptive_window)#_adaptive(signal, window=adaptive_window)
    if use_pyampd:
        peaks = _pyampd(signal, adaptive_window, min_prominence = min_prominence, **kwargs)
    else:
        peaks,_ = find_peaks(signal, prominence = min_prominence, **kwargs)
    if min_distance is not None:
        prominence, frac_illegal = _illegal_intervals(signal, peaks, min_distance)
        peaks = _select_valid_peaks(peaks, prominence, frac_illegal, sensitivity)
    return peaks


def calculate_time(df, fps):
    """calculate time from frame index."""
    # real time
    df.loc[:,'time'] = df['frame']/fps
    return df
    
    
def calculate_locations(df, scale):
    """calculate correctly scaled x,y coordinates."""
    # real time
    df.loc[:,'x_scaled'] = df['x']*scale
    df.loc[:,'y_scaled'] = df['y']*scale
    return df
    

def calculate_velocity(df, scale, fps, dt = 1):
    """calculate velocity from the coordinates."""
    cms = np.stack([df.x, df.y]).T
    v_cms = cms[dt:]-cms[:-dt]
    t = np.array(df.frame)
    deltat = t[dt:]-t[:-dt]
    velocity = np.sqrt(np.sum((v_cms)**2, axis = 1))/deltat*scale*fps
    velocity = np.append(velocity, [np.nan]*dt)
    df.loc[:,'velocity'] = velocity
    return df


def calculate_pumps(df, min_distance, sensitivity, adaptive_window, min_prominence = 0.2, key = 'pump_clean', use_pyampd = True, fps=30):
    """using a pump trace, get additional pumping metrics."""
    signal = df[key]
    peaks = detect_peaks(signal, adaptive_window, min_distance, min_prominence, sensitivity, use_pyampd)
    if len(peaks)>1:
        # add interpolated pumping rate to dataframe
        df['rate'] = np.interp(np.arange(len(df)), peaks[:-1], fps/np.diff(peaks))
        # # get a binary trace where pumps are 1 and non-pumps are 0
        events= np.zeros(len(df['rate']))
        events[peaks] = 1
        df.loc[:,'pump_events'] = events
    else:
        df.loc[:,'rate'] = 0
        df.loc[:,'pump_events'] = 0
    return df



def calculate_reversals_nose(df, dt =1, angle_threshold = 150, w_smooth = 30, min_duration = 30):
    """using the motion of the nosetip relative to the center of mass motion to determine reversals."""
    cl = np.array([np.array(cl) for cl in df['Centerline']])
    
    cms = np.stack([df.x, df.y]).T
    # trajectories - CMS - coarse-grain
    # check the number of points of each centerline
    nPts = len(cl[0])
    # note, the centerline is ordered y,x
    # subtract the mean since the cl is at (50,50) or similar
    yc, xc = cl.T - np.mean(cl.T, axis = 1)[:,np.newaxis]
    cl_new = np.stack([xc, yc]).T + np.repeat(cms[:,np.newaxis,:], nPts, axis = 1)
    nose = cl_new[:,0,:]
    #calculate directions - in the lab frame! and with dt
    v_nose = nose[dt:]-nose[:-dt]
    #v_cms = cms[dt:]-cms[:-dt]
    # tangent of the worm = 'heading'
    heading = cl_new[:,0,]-cl_new[:,nPts//2,]

    # angle relative to cms motion
    angle = np.array([np.arccos(np.dot(v1, v2)/(np.linalg.norm(v2)*np.linalg.norm(v1)))*180/np.pi for \
                      (v1, v2) in zip(v_nose, heading)])
    angle = pd.Series(angle).rolling(w_smooth, min_periods=1, center=True).apply(circmean, args=(180, 0))
    angle[np.isnan(angle)] = 0
    angle = np.append(angle, [np.nan]*dt)
    
    df.loc[:,'angle_nose'] = angle
    # determine when angle is over threshold
    rev = angle > angle_threshold
    # filter short reversals
    rev = pd.Series(rev).rolling(2*min_duration, center=True).median()
    #rev = np.append(rev, [np.nan]*dt)
    #df.add_column('reversals_nose', rev, overwrite = True)
    df['reversals_nose'] = rev
    # add reversal events (1 where a reversal starts)
    reversal_start = np.diff(np.array(rev, dtype=int))==1
    reversal_start = np.append(reversal_start, [0])
    #df.add_column('reversal_events_nose', reversal_start, overwrite = True)
    df.loc[:,'reversal_events_nose'] = reversal_start
    return df


def calculate_reversals(df, animal_size, angle_threshold, scale):
    """Adaptation of the Hardaker's method to detect reversal event. 
    A single worm's centroid trajectory is re-sampled with a distance interval equivalent to 1/10 
    of the worm's length (100um) and then reversals are calculated from turning angles.
    Inputs:
        animal_size: animal size in um.
        angle_threshold (degree): what defines a turn 
    Output: None, but adds a column 'reversals' to  df.
    """
    # resampling
    # Calculate the distance cover by the centroid of the worm between two frames um
    distance = np.cumsum(df['velocity'])*df['time'].diff()
    # find the maximum number of worm lengths we have travelled
    maxlen = distance.max()/animal_size
    # make list of levels that are multiples of animal size
    levels = np.arange(animal_size, maxlen*animal_size, animal_size)
    # Find the indices where the distance is equal or the closest to the pixel interval by repeatedly subtracting the levels
    indices = []
    for level in levels:
        idx = distance.sub(level).abs().idxmin()
        indices.append(idx)
    # create a downsampled trajectory from these indices
    traj_Resampled = df.loc[indices, ['x', 'y']].diff()*scale
    # we ignore the index here for the shifted data
    traj_Resampled[['x1', 'y1']] = traj_Resampled.shift(1).fillna(0)
    # use the dot product to calculate the andle
    def angle(row):
        v1 = [row.x, row.y]
        v2 = [row.x1, row.y1]
        return np.degrees(np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
    traj_Resampled['angle'] = traj_Resampled.apply(lambda row: angle(row), axis =1)
    rev = traj_Resampled.index[traj_Resampled.angle>=angle_threshold]
    df.loc[:,'reversals'] = 0
    df.loc[rev,'reversals'] = 1
    return df
 
