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
    return filters.threshold_li(img)


@pims.pipeline
def preprocess(img, minSize = 800, threshold = None, smooth = 0):
    """
    Apply image processing functions to return a binary image. 
    """
    # smooth
    if smooth:
        img = filters.gaussian(img, smooth, preserve_range = True)
    # Apply thresholds
    if threshold ==None:
        threshold = filters.threshold_yen(img)#, initial_guess=skimage.filters.threshold_otsu)
    #mask = filters.rank.threshold(img, morphology.square(3), out=None, mask=None, shift_x=False, shift_y=False)
    mask = img >= threshold
    # dilations
    mask = morphology.remove_small_holes(mask, area_threshold=4, connectivity=1, in_place=True)
    #mask = ndimage.binary_dilation(mask)
    #mask = morphology.remove_small_objects(mask, min_size=minSize, connectivity=1, in_place=True)
    return mask


@pims.pipeline
def refine(img, size):
    """Refine segmentation using watershed."""
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((size, size)),\
                                labels=img)
    markers = ndi.label(local_maxi)[0]
    return watershed(-distance, markers, mask=img)


def calculateMask(frames, bgWindow = 15, thresholdWindow = 30, minSize = 50, subtract = False, **kwargs):
    """standard median stack-projection to obtain a background image followd by thresholding and filtering of small objects to get a clean mask."""
    if subtract:
        bg = np.median(frames[::bgWindow], axis=0)
        #subtract bg from all frames
        frames = subtractBG(frames, bg)
    # get an overall threshold value and binarize images by using z-stack

    thresh = getThreshold(np.max(frames[::thresholdWindow], axis=0))
    #threshs = getThreshold(frames[::thresholdWindow])
    #thresh = np.median(threshs)
    return preprocess(frames, minSize, threshold = thresh, **kwargs)



def objectDetection( mask, img, params, frame):
    """label binary image and extract a region of interest around the object."""
    df = pd.DataFrame()
    label_image = skimage.measure.label(mask, background=0, connectivity = 1)
    label_image = skimage.segmentation.clear_border(label_image, buffer_size=0, bgval=0, in_place=False, mask=None)
    for region in skimage.measure.regionprops(label_image, intensity_image=img):
        if region.area > params['minSize'] and region.area < params['maxSize']:
            # get a larger than bounding box image by padding some
            p = int(params['pad'])
            xmin, ymin, xmax, ymax  = region.bbox
            sliced = slice(np.max([0, xmin-p]), xmax+p), slice(np.max([0, ymin-p]), ymax+p)
            im = img[sliced]
            # Store features which survived to the criterions
            df = df.append([{'y': region.centroid[0],
                             'x': region.centroid[1],
                             'slice':region.slice,
                             'frame': frame,
                             'area': region.area,
                             'image': im,#region.intensity_image,
                             'yw': region.weighted_centroid[0],
                             'xw': region.weighted_centroid[1]
                             },])
        # do watershed to get crossings separated. 
        elif region.area > params['minSize']:
            image = mask[region.slice]
            labeled = refine(image, size = params['watershed'])
            for part in skimage.measure.regionprops(labeled, intensity_image=img[region.slice]):
                if part.area > params['minSize'] and part.area < 1.5*params['maxSize']:
                    # get a larger than bounding box image by padding some
                    p = int(params['pad'])
                    xmin, ymin, xmax, ymax  = part.bbox
                    sliced = slice(np.max([0, xmin-p]), xmax+p), slice(np.max([0, ymin-p]), ymax+p)
                    im = img[sliced]
                    # Store features which survived to the criterions
                    df = df.append([{'y': part.centroid[0],
                                     'x': part.centroid[1],
                                     'slice':part.slice,
                                     'frame': frame,
                                     'area': part.area,
                                     'image': im,#region.intensity_image,
                                     'yw': part.weighted_centroid[0],
                                     'xw': part.weighted_centroid[1]
                                     },])
    return df


def runfeatureDetection(frames, masks, params, frameOffset):
    """detect objects in each image and use region props to extract features and store a local image."""
    feat = []
    for num, img in enumerate(frames):
        feat.append(objectDetection(masks[num], img, params, num+frameOffset))
    features = pd.concat(feat)
    return features


def linkParticles(df, searchRange, minimalDuration, **kwargs):
    """input is a pandas dataframe that contains at least the columns 'frame' and 'x', 'y'. the function
    inplace modifies the input dataframe by adding a column called 'particles' which labels the objects belonging to one trajectory. 
    **kwargs can be passed to the trackpy function link_df to modify tracking behavior."""
    traj = tp.link_df(df, searchRange, **kwargs)
    # filter short trajectories
    traj = tp.filter_stubs(traj, minimalDuration)
    # make a numerical index
    traj.set_index(np.arange(len(traj.index)), inplace = True)
    return traj