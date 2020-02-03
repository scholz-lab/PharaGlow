#!/usr/bin/env python

"""run.py: run pharaglow analysis by inplace modifying pandas dataframes."""

import numpy as np
from skimage.io import imread
from skimage import util
import json
import pandas as pd

import pharaglow.features as pg
import pharaglow.tracking as pgt
import pharaglow.util as pgu


def runPharaglowSkel(im, length):
    # preprocessing image
    im = pgu.unravelImages(im, length)
    im = np.array(im)
    mask = pg.thresholdPharynx(im)
    skel = pg.skeletonPharynx(mask)
    order = pg.sortSkeleton(skel)
    ptsX, ptsY = np.where(skel)
    ptsX, ptsY = ptsX[order], ptsY[order]
    return mask.ravel(), ptsX, ptsY


def runPharaglowCL(mask, ptsX, ptsY, length, nPts = 100):
    # mask reshaping from linear
    mask = pgu.unravelImages(mask, length)
    # getting centerline and widths along midline
    poptX, poptY = pg.fitSkeleton(ptsX, ptsY)
    contour = pg.morphologicalPharynxContour(mask, scale = 4, smoothing=2)
    xstart, xend = pg.cropcenterline(poptX, poptY, contour, nP = len(ptsX))
    # extend a bit to include all bulbs
    #l = np.abs(xstart-xend)
    #xstart -= 0.05*l
    #xend += 0.05*l
    # make sure centerline isn't too short if something goes wrong
    if np.abs(xstart-xend) < 0.5*len(ptsX):
        xstart, xend = 0, len(ptsX)
    xs = np.linspace(xstart, xend, nPts)
    cl = pg.centerline(poptX, poptY, xs)
    dCl = pg.normalVecCl(poptX, poptY, xs)
    widths = pg.widthPharynx(cl, contour, dCl)
    if (np.argmax(pg.scalarWidth(widths)) - 0.5*len(widths)) <= 0:
        xtmp = xstart
        xstart = xend
        xend = xtmp
        cl = cl[::-1]
        dCl = dCl[::-1]
        widths = widths[::-1]
    return poptX, poptY, xstart, xend, cl, dCl, widths, contour


def runPharaglowKymo(im, cl, widths, length, **kwargs):
    im = pgu.unravelImages(im, length)
    im = np.array(im)
    kymo = pg.intensityAlongCenterline(im, cl, **kwargs)
    kymoWeighted = pg.intensityAlongCenterline(im, cl, width = pg.scalarWidth(widths))[:,0]
    return kymo, kymoWeighted


def runPharaglowImg(im, xstart, xend, poptX, poptY, width, npts, length):
    # make sure image is float
    im = pgu.unravelImages(im, length)
    im = np.array(im)
    #im = util.img_as_float64(im)
    #local derivative, can enhance contrast
    gradientImage = pg.gradientPharynx(im)

    # straightened image
    straightIm = pg.straightenPharynx(im, xstart, xend, poptX, poptY, width=width, nPts = npts)
    return gradientImage, straightIm


def pharynxorientation(df):
    """get the orientation from the minimal trajectory of the start and end points."""
    df.loc[:,'StraightKymo'] = df.apply(
    lambda row: np.mean(row['Straightened'], axis = 1), axis=1)

    df['Similarity'] = False
    for particleID in df['particle'].unique():
        mask = df['particle']==particleID
        sample = df[mask]['StraightKymo'].iloc[0]
        # let's make sure the sample is anterior to posterior
        #if np.mean(sample[:len(sample//2)])<np.mean(sample[len(sample//2):]):
        #    sample = sample[::-1]
        df['Similarity'] = np.where(mask, df.apply(\
          lambda row: np.sum((row['StraightKymo']-sample)**2)<np.sum((row['StraightKymo']-sample[::-1])**2), axis=1)\
                                    , df['Similarity'])

    # now flip the orientation where the sample is upside down
    for key in ['SkeletonX', 'SkeletonY', 'Centerline', 'dCl', 'Widths', 'Kymo',\
           'WeightedKymo', 'StraightKymo', 'Straightened', 'KymoGrad', 'WeightedKymoGrad']:
        df[key] = df.apply(lambda row: row[key] if row['Similarity'] else row[key][::-1], axis=1)
    # Flip the start coordinates
    #df.update(df.loc[df['Similarity']].rename({'Xstart': 'Xend', 'Xend': 'Xstart'}, axis=1))
    df['Xtmp'] = df['Xstart']
    df['Xstart'] = df.apply(lambda row: row['Xstart'] if row['Similarity'] else row['Xend'], axis=1)
    df['Xend'] = df.apply(lambda row: row['Xend'] if row['Similarity'] else row['Xtmp'], axis=1)
    return df


def runPharaglowOnStack(df, param):
    """runs the whole pharaglow toolset on a dataframe. Can be linked or unliked before. only the last step depends on having a particle ID."""
    # run image analysis on all rows of the dataframe
    df[['Mask', 'SkeletonX', 'SkeletonY']] = df.apply(\
        lambda row: pd.Series(runPharaglowSkel(row['image'], param['length'])), axis=1)
    # run centerline fitting
    df[['ParX', 'ParY', 'Xstart', 'Xend', 'Centerline', 'dCl', 'Widths', 'Contour']] = df.apply(\
        lambda row: pd.Series(runPharaglowCL(row['Mask'],row['SkeletonX'], row['SkeletonY'], param['length'])), axis=1)
    # run image operations
    df[['Gradient', 'Straightened']] = df.apply(\
        lambda row: pd.Series(runPharaglowImg(row['image'], row['Xstart'], row['Xend'],\
                                              row['ParX'], row['ParY'], param['widthStraight'],\
                                             param['nPts'], param['length'])), axis=1)
    # run kymographs
    df[['Kymo', 'WeightedKymo']] = df.apply(\
        lambda row: pd.Series(runPharaglowKymo(row['image'], row['Centerline'], row['Widths'], params['length'],params['linewidth'])), axis=1)
    # run kymographs
    df[['KymoGrad', 'WeightedKymoGrad']] = df.apply(\
        lambda row: pd.Series(runPharaglowKymo(row['Gradient'], row['Centerline'], row['Widths'], params['length'], params['linewidth'])), axis=1)
    ## clean orientation
    df = pharynxorientation(df)
    return df



# def main():
#     fname = "/home/scholz_la/Dropbox (Scholz Lab)/Scholz Lab's shared workspace/Nicolina ImageJ analysis/NZ_0007_croppedsample.tif"
#     parameterfile = "/home/scholz_la/Dropbox (Scholz Lab)/Scholz Lab's shared workspace/Nicolina ImageJ analysis/pharaglow_parameters.txt"
#     print('Starting pharaglow analysis...')
#     rawframes = pims.open(fname)
#     print('Analyzing', rawframes)
#     print('Loading parameters from {}'.format(parameterfile))
#     with open(parameterfile) as f:
#         param = json.load(f)
    
#     print('Binarizing images')
#     masks = pgt.calculateMasks(rawframes)
#     print('Detecting features')
#     features = pgt.runfeatureDetection(frames, masks)
#     print('Linking trajectories')
#     trajectories = pgt.linkParticles(features, param['searchRange'], param['minimalDuration'], **kwargs)
#     print('Extracting pharynx data')
#     trajectories = runPharaglowOnStack(trajectories)
#     print('Done tracking. Successfully tracked {} frames with {} trajectories.'.format(len(frames), trajectories['particle'].nunique()))