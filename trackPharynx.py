import matplotlib.pylab as plt
import numpy as np
from skimage.io import imread
import json

import pharaglow.features as pg
import pharaglow.tracking as pgt


def runPharaglowSkel(im):
    # preprocessing image
    mask = pg.thresholdPharynx(im)
    skel = pg.skeletonPharynx(mask)
    order = pg.sortSkeleton(skel)
    ptsX, ptsY = np.where(skel)
    ptsX, ptsY = ptsX[order], ptsY[order]
    return mask, ptsX, ptsY


def runPharaglowCL(mask, ptsX, ptsY, nPts = 25):
    # getting centerline and widths along midline
    poptX, poptY = pg.fitSkeleton(ptsX, ptsY)
    contour = pg.morphologicalPharynxContour(mask, scale = 4, smoothing=2)
    xstart, xend = pg.cropcenterline(poptX, poptY, contour, nP = len(ptsX))
    # make sure centerline isn't too short if something goes wring
    if np.abs(xstart-xend) < 0.5*len(ptsX):
        xstart, xend = 0, len(ptsX)
    xs = np.linspace(xstart, xend, nPts)
    cl = pg.centerline(poptX, poptY, xs)
    dCl = pg.normalVecCl(poptX, poptY, xs)
    widths = pg.widthPharynx(cl, contour, dCl)

    return poptX, poptY, xstart, xend, cl, dCl, widths, contour


def runPharaglowKymo(im, cl, widths, **kwargs):
    kymo = pg.intensityAlongCenterline(im, cl, **kwargs)
    kymoWeighted = pg.intensityAlongCenterline(im, cl, width = pg.scalarWidth(widths))
    return kymo, kymoWeighted


def runPharaglowImg(im, xstart, xend, poptX, poptY, widths):
    # make sure image is float
    im = util.img_as_float64(im)
    #local derivative, can enhance contrast
    gradientImage = pg.gradientPharynx(im)
    # straightened image
    w = np.max(pg.scalarWidth(widths))//2
    straightIm = pg.straightenPharynx(im, xstart, xend, poptX, poptY, width=8)
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
        df['Similarity'] = np.where(mask, df.apply(
          lambda row: np.sum((row['StraightKymo']-sample)**2)>np.sum((row['StraightKymo']-sample[::-1])**2), axis=1)\
                                    , df['Similarity'])

    # now flip the orientation where the sample is upside down
    for key in ['SkeletonX', 'SkeletonY', 'Centerline', 'dCl', 'Widths', 'Kymo',
           'WeightedKymo', 'StraightKymo', 'Straightened']:
        df[key] = df.apply(lambda row: row[key] if row['Similarity'] else row[key][::-1], axis=1)
    # Flip the start coordinates
    #df.update(df.loc[df['Similarity']].rename({'Xstart': 'Xend', 'Xend': 'Xstart'}, axis=1))
    df['Xtmp'] = df['Xstart']
    df['Xstart'] = df.apply(lambda row: row['Xstart'] if row['Similarity'] else row['Xend'], axis=1)
    df['Xend'] = df.apply(lambda row: row['Xend'] if row['Similarity'] else row['Xtmp'], axis=1)


def runPharaglowOnStack(df):
    """runs the whole pharaglow toolset on a dataframe. Can be linked or unliked before. only the last step depends on having a particle ID."""
    # run image analysis on all rows of the dataframe
    df[['Mask', 'SkeletonX', 'SkeletonY']] = df.apply(
        lambda row: pd.Series(runPharaglowSkel(row['image'])), axis=1)
    # run centerline fitting
    df[['ParX', 'ParY', 'Xstart', 'Xend', 'Centerline', 'dCl', 'Widths', 'Contour']] = df.apply(
        lambda row: pd.Series(runPharaglowCL(row['Mask'],row['SkeletonX'], row['SkeletonY'])), axis=1)
    # run image operations
    df[['Gradient', 'Straightened']] = df.apply(
        lambda row: pd.Series(runPharaglowImg(row['image'], row['Xstart'], row['Xend'],\
                                              row['ParX'], row['ParY'], row['Widths'],\
                                             )), axis=1)
    # run kymographs
    df[['Kymo', 'WeightedKymo']] = df.apply(
        lambda row: pd.Series(runPharaglowKymo(row['image'], row['Centerline'], row['Widths'], linewidth = 2)), axis=1)
    ## clean orientation
    pharynxorientation(df)
    return df



def main():
    fname = "/home/scholz_la/Dropbox (Scholz Lab)/Scholz Lab's shared workspace/Nicolina ImageJ analysis/NZ_0007_croppedsample.tif"
    parameterfile = "/home/scholz_la/Dropbox (Scholz Lab)/Scholz Lab's shared workspace/Nicolina ImageJ analysis/pharaglow_parameters.txt"
    print('Starting pharaglow analysis...')
    rawframes = pims.open(fname)
    print('Analyzing', rawframes)
    print('Loading parameters from {}'.format(parameterfile))
    with open(parameterfile) as f:
        param = json.load(f)
    
    print('Binarizing images')
    masks = pgt.calculateMasks(rawframes)
    print('Detecting features')
    features = pgt.runfeatureDetection(frames, masks)
    print('Linking trajectories')
    trajectories = pgt.linkParticles(features, param['searchRange'], param['minimalDuration'], **kwargs)
    print('Extracting pharynx data')
    trajectories = runPharaglowOnStack(trajectories)
    print('Done tracking. Successfully tracked {} frames with {} trajectories.'.format(len(frames), trajectories['particle'].nunique()))