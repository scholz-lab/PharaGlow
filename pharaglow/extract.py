#!/usr/bin/env python

"""extract.py: Extract pumping-like traces from kymographs etc.."""
import pandas as pd
import numpy as np
from scipy.stats import skew

import pharaglow.features as pg

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