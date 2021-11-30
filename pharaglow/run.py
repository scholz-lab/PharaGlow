#!/usr/bin/env python

"""run.py: run pharaglow analysis by inplace modifying pandas dataframes."""

import numpy as np
from skimage.io import imread
import pandas as pd

import pharaglow.features as pg
from . import util as pgu


def runPharaglowSkel(im):
    """ Create a centerline of the object in the image by binarizing,
    skeletonizing and sorting centerline points.

    Args:
        im (numpy.array or pims.Frame): image

    Returns:
        list: binary of image, unraveled
        list: coordinates of centerline along X
        list: coordinates of centerline along Y
    """

    mask = pg.thresholdPharynx(im)
    skel = pg.skeletonPharynx(mask)
    # if the image is empty or nearly empty
    if np.sum(mask) == 0 or np.sum(skel)<6:
        return mask.ravel(), np.nan, np.nan
    order = pg.sortSkeleton(skel)
    ptsX, ptsY = np.where(skel)
    ptsX, ptsY = ptsX[order], ptsY[order]
    return mask.ravel(), ptsX, ptsY


def runPharaglowCL(mask, ptsX, ptsY, length, **kwargs):
    """ Fit the centerline points and detect object morphology.

    Args:
        mask (list): binary image, unraveled
        ptsX (list): coordinates of centerline along X
        ptsY (list): coordinates of centerline along Y
        length (list): length of one axis of the image

    Returns:
        list: poptX - optimal fit parameters of .features.pharynxFunc
        list: poptY - optimal fit parameters of .features.pharynxFunc
        float: xstart -start coordinate to apply to .features._pharynxFunc(x) to create a centerline
        float: xend - end coordinate to apply to .features._pharynxFunc(x) to create a centerline
        list: cl - (n,2) list of centerline coordinates in image space.
        list: dCl - (n,2) array of unit vectors orthogonal to centerline. Same length as cl.
        list: widths - (n,2) widths of the contour at each centerline point.
        list: contour- (m,2) coordinates along the contour of the object
    """
    # mask reshaping from linear
    mask = pgu.unravelImages(mask, length)
    # getting centerline and widths along midline
    poptX, poptY = pg.fitSkeleton(ptsX, ptsY)
    scale=  kwargs.pop('scale', 4)
    contour = pg.morphologicalPharynxContour(mask, scale)
    xstart, xend = pg.cropcenterline(poptX, poptY, contour)
    # create uniform spacing along line
    xs = np.linspace(xstart, xend, 100)
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


def runPharaglowKymo(im, cl, widths, **kwargs):
    """ Use the centerline to extract intensity along this line from an image.

    Args:
        im (numpy.array): image of a pharynx
        cl (numpy.array or list): (n,2) list of centerline coordinates in image space.
        kwargs: **kwargs are passed skimage.measure.profile_line.

    Returns:
        numpy.array: array of (?,) length. Length is determined by pathlength of centerline.
    """
    #kymoWeighted = pg.intensityAlongCenterline(im, cl, width = pg.scalarWidth(widths))[:,0]
    return [pg.intensityAlongCenterline(im, cl, **kwargs)]


def runPharaglowImg(im, xstart, xend, poptX, poptY, width, npts):
    """ Obtain the straightened version and gradient of the input image.

    Args:
        im (numpy.array or pims.Frame): image of curved object
        xstart (float): start coordinate to apply to .features._pharynxFunc(x) to
                create a centerline
        xend (float):  end coordinate to apply to .features._pharynxFunc(x)
                to create a centerline
        poptX (array): optimal fit parameters describing pharynx centerline.
        poptY (array): optimal fit parameters describing pharynx centerline.
        width (int): how many points to sample orthogonal of the centerline
        nPts (int, optional): how many points to sample along the centerline. Defaults to 100.

    Returns:
        numpy.array: local derivative of image
        numpy.array:  (nPts, width) array of image intensity straightened by centerline
    """
    #local derivative, can enhance contrast
    gradientImage = pg.gradientPharynx(im)
    # straightened image
    straightIm = pg.straightenPharynx(im, xstart, xend, poptX, poptY, width=width, nPts = npts)
    return gradientImage, straightIm


def pharynxorientation(df):
    """ Get all images into the same orientation by comparing to a sample image.

    Args:
        df (pandas.DataFrame): a pharaglow dataframe after running .run.runPharaglowOnImage()

    Returns:
        pandas.DataFrame: dataFrame with flipped columns where neccessary
    """
    df.loc[:,'StraightKymo'] = df.apply(
    lambda row: np.mean(row['Straightened'], axis = 1), axis=1)
    df['Similarity'] = False
    sample = df['StraightKymo'].mean()
    # let's make sure the sample is anterior to posterior
    if np.mean(sample[:len(sample//2)])>np.mean(sample[len(sample//2):]):
        sample = sample[::-1]
    # this uses the intenisty profile - sometimes goes wrong
    df['Similarity'] = df.apply(\
        lambda row: np.sum((row['StraightKymo']-sample)**2)<\
            np.sum((row['StraightKymo']-sample[::-1])**2), axis=1)
    # now flip the orientation where the sample is upside down
    for key in ['SkeletonX', 'SkeletonY', 'Centerline', 'dCl', 'Widths', 'Kymo',\
           'StraightKymo', 'Straightened', 'KymoGrad']:
        if key in df.columns:
            df[key] = df.apply(lambda row: row[key] if row['Similarity'] \
                else row[key][::-1], axis=1)
    # Flip the start coordinates
    if set(['Xstart', 'Xend']).issubset(df.columns):
        df['Xtmp'] = df['Xstart']
        df['Xstart'] = df.apply(lambda row: row['Xstart'] if \
            row['Similarity'] else row['Xend'], axis=1)
        df['Xend'] = df.apply(lambda row: row['Xend'] if row['Similarity'] else row['Xtmp'], axis=1)
    return df


def runPharaglowOnImage(image, framenumber, params, **kwargs):
    """ Run pharaglow-specific image analysis of a pharynx on a single image.

    Args:
        image (numpy.array or pims.Frame): input image
        framenumber (int): frame number to indicate in the resulting
        dataframe which image is being analyzed.
        arams (dict): parameter dictionary containing image analysis parameters.

    Returns:
        pandas.DataFrame: collection of data created by pharaglow for this image.
    """

    if 'run_all' in kwargs.keys():
        run_all = kwargs['run_all']
    else:
        run_all = False
    colnames = ['Mask', 'SkeletonX', 'SkeletonY','ParX', 'ParY', 'Xstart',\
         'Xend', 'Centerline', 'dCl', 'Widths', \
        'Contour','Gradient', 'Straightened', 'Kymo', 'KymoGrad']
    # skeletonize
    mask, skelX, skelY = runPharaglowSkel(image)
    if np.sum(mask) == 0 or np.any(np.isnan(skelX)) or np.any(np.isnan(skelY)):
        results = np.ones(len(colnames))*np.nan
    else:
        #centerline fit
        scale = params.pop('scale', 4)
        parX, parY, xstart, xend, cl, dCl, widths, contour = \
            runPharaglowCL(mask,skelX, skelY, params['length'], scale = scale)
        # image transformation operations
        grad, straightened = runPharaglowImg(image, xstart,xend,\
                                            parX, parY, params['widthStraight'],\
                                            params['nPts'])
        results = [mask, skelX, skelY, parX, parY, xstart, xend,\
             cl, dCl, widths, contour, grad, straightened]
        if run_all:
            # run kymographs
            kymo= runPharaglowKymo(image, cl, widths, linewidth = params['linewidth'])
            # run kymographs
            kymograd = runPharaglowKymo(grad, cl, widths, linewidth = params['linewidth'])
            results.append(kymo, kymograd)
    data = {}
    for col, res in zip(colnames,results):
        data[col] = res
    df = pd.DataFrame([data], dtype = 'object')
    df['frame'] = framenumber
    #print('Done', framenumber)
    return df,


def parallel_pharaglow_run(args, **kwargs):
    """ Define a worker function for parallelization.

    Args:
        args (div.): arguments for .features.runPharaglowOnImage()

    Returns:
        pandas.DataFrame: hands over output from .features.runPharaglowOnImage()
    """

    return runPharaglowOnImage(*args, **kwargs)
