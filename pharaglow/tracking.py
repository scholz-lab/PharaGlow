#!/usr/bin/env python

"""tracking.py: trackpy based worm tracking."""

import numpy as np
import pandas as pd

import pims
import trackpy as tp
import os
import skimage
from skimage.measure import label
# preprocessing the image for tracking. 
from skimage import morphology, util, filters, segmentation
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

@pims.pipeline
def subtractBG(img, bg):
    """
    Subtract a background from the image.
    """
    tmp = img-bg
    tmp -= np.min(tmp)
    tmp /=np.max(tmp)
    return util.img_as_float(tmp)


@pims.pipeline
def getThreshold(img):
    """"return a global threshold value"""
    return filters.threshold_yen(img)


@pims.pipeline
def preprocess(img, minSize = 800, threshold = None):
    """
    Apply image processing functions to return a binary image
    """
    # smooth
    img = filters.gaussian(img, 2)
    # Apply thresholds
    if threshold ==None:
        threshold = filters.threshold_yen(img)#, initial_guess=skimage.filters.threshold_otsu)
    #mask = filters.rank.threshold(img, morphology.square(3), out=None, mask=None, shift_x=False, shift_y=False)
    mask = img >= threshold
    # dilations
    #mask = ndimage.binary_dilation(mask)
    #mask = ndimage.binary_dilation(mask)
    mask = morphology.remove_small_objects(mask, min_size=minSize, connectivity=1, in_place=True)
    return mask


@pims.pipeline
def refine(img):
    """Refine segmentation using watershed."""
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((5, 5)),\
                                labels=img)
    markers = ndi.label(local_maxi)[0]
    return watershed(-distance, markers, mask=img)


def calculateMask(frames, bgWindow = 15, thresholdWindow = 30, minSize = 50 ):
    """standard median stack-projection to obtain a background image followd by thresholding and filtering of small objects to get a clean mask."""
    bg = np.median(frames[::bgWindow], axis=0)
    #subtract bg from all frames
    frames = subtractBG(frames, bg)
    # get an overall threshold value and binarize images 
    threshs = getThreshold(frames[::thresholdWindow])
    thresh = np.median(threshs)
    return preprocess(frames, minSize, threshold = thresh)



def runfeatureDetection(frames, masks):
    """detect objects in each image and use region props to extract features and store a local image."""
    features = pd.DataFrame()
    for num, img in enumerate(frames):
        label_image = skimage.measure.label(masks[num], background=0, connectivity = 2)
        for region in skimage.measure.regionprops(label_image, intensity_image=img):
            if region.area > 200:
                # Store features which survived to the criterions
                features = features.append([{'y': region.centroid[0],
                                             'x': region.centroid[1],
                                             'slice':region.slice,
                                             'frame': num,
                                             'area': region.area,
                                             'image': region.intensity_image
                                             },])
    return features


def linkParticles(df, searchRange, minimalDuration, **kwargs):
    """input is a pandas dataframe that contains at least the columns 'frame' and 'x', 'y'. the function
    inplace modifies the input dataframe by adding a column called 'particles' which labels the objects belonging to one trajectory. 
    **kwargs can be passed to the trackpy function link_df to modify tracking behavior."""
    traj = tp.link_df(df, searchRange)
    # filter short trajectories
    tp.filter_stubs(traj, minimalDuration)
    # make a numerical index
    traj.reset_index(inplace=True)
    return traj