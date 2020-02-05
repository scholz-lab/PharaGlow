#!/usr/bin/env python

"""tracking.py: trackpy based worm tracking."""

import numpy as np
import pandas as pd
import warnings

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
    mask = morphology.remove_small_holes(mask, area_threshold=12, connectivity=1, in_place=True)
    mask = ndi.binary_dilation(mask)
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


def extractImage(img, mask, length, cmsLocal):
    """extracts a square image of an object centered around center of mass coordinates with size (length, length). Mask ensures that 
    only one object is visible if two are in the same region.
    img is the bounding box region of the object."""
    assert length%2==0, "length should be an even number to rounding artefacts."
    im = np.zeros((length, length))
    yc, xc = np.rint(cmsLocal).astype(np.int32)
    sy, sx = img.shape
    # check that the image will fit in the bounded region
    if sx>=length:
        warnings.warn('The object is larger than the bounding box. \
            Try increasing the length parameter.', Warning)
        img = img[:,xc - length//2:xc + length//2]
        mask = mask[:,xc - length//2:xc + length//2]
        xc = length//2
    if sy>=length:
        warnings.warn('The object is larger than the bounding box. \
            Try increasing the length parameter.', Warning)
        img = img[yc - length//2:yc + length//2]
        mask = mask[yc - length//2:yc + length//2]
        yc = length//2
    sy, sx = img.shape
    assert sx<=length and sy<=length, "The size of the object is larger than the bounding box. \
            Try increasing the length parameter. (object: {}, length: {})".format(np.max([sx,sy]), length)

    yoff = length//2-yc
    xoff = length//2-xc
    # check that the image will fit in the bounded region
    assert yoff>=0 and xoff>=0, "The size of the object is larger than the bounding box. \
            Try increasing the length parameter."
    im[yoff:yoff+sy, xoff:xoff+sx] = img*mask
    return im


def objectDetection(mask, img, params, frame):
    """label binary image and extract a region of interest around the object."""
    df = pd.DataFrame()
    label_image = skimage.measure.label(mask, background=0, connectivity = 1)
    label_image = skimage.segmentation.clear_border(label_image, buffer_size=0, bgval=0, in_place=False, mask=None)
    for region in skimage.measure.regionprops(label_image, intensity_image=img):
        if region.area > params['minSize'] and region.area < params['maxSize']:
            # get the image of an object
            im = extractImage(region.intensity_image, region.image, params['length'], region.local_centroid)
            # Store features which survived to the criterions
            df = df.append([{'y': region.centroid[0],
                             'x': region.centroid[1],
                             'slice':region.slice,
                             'frame': frame,
                             'area': region.area,
                             'image': im.ravel(),
                             'yw': region.weighted_centroid[0],
                             'xw': region.weighted_centroid[1]
                             },])
        # do watershed to get crossing objects separated. 
        elif region.area > params['minSize']:
            image = mask[region.slice]
            labeled = refine(image, size = params['watershed'])
            for part in skimage.measure.regionprops(labeled, intensity_image=img[region.slice]):
                if part.area > params['minSize'] and part.area < 1.1*params['maxSize']:
                    # get the image of an object
                    im = extractImage(part.intensity_image, part.image, params['length'], part.local_centroid)
                    # Store features which survived to the criterions
                    df = df.append([{'y': part.centroid[0],
                                     'x': part.centroid[1],
                                     'slice':part.slice,
                                     'frame': frame,
                                     'area': part.area,
                                     'image': im.ravel(),
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


def interpolateTrajectories(traj):
    """given a dataframe with a trajectory, interpolate missing frames.
    The interpolate function ignores non-pandas types, so no image interpolations."""
    idx = pd.Index(np.arange(traj['frame'].min(), traj['frame'].max()), name="frame")
    traj = traj.set_index("frame").reindex(idx).reset_index()
    return traj.interpolate()


def cropImagesAroundCMS(img, x, y, length, size):
    """Using the interpolated center of mass coordindates (x,y), fill in missing images. img is a full size frame."""
    xmin, xmax = int(x - length//2), int(x + length//2)
    ymin, ymax = int(y-length//2), int(y+length//2)
    sliced = slice(np.max([0, ymin]), ymax), slice(np.max([0, xmin]), xmax)
    im = img[sliced]
    padx = [np.max([-xmin, 0]), np.max([xmax-img.shape[1], 0])]
    pady = [np.max([-ymin, 0]), np.max([ymax-img.shape[0], 0])]
    im = np.pad(im, [pady, padx] , mode='constant')
    # refine to a single animal if neccessary
    mask = preprocess(im, minSize = 0, threshold = None, smooth = 0)
    labeled = refine(mask, size)
    d = length//2
    if len(np.unique(labeled))>=2:
        for part in skimage.measure.regionprops(labeled):
            d2 = np.sqrt((part.centroid[0]-length//2)**2+(part.centroid[1]-length//2)**2)
            if d2 < d:
                mask = labeled==part.label
    return (im*mask).ravel()


def fillMissingImages(imgs, frame, x, y, length, size):
    """run this on a dataframe to interpolate images from missing coordinates."""
    img = imgs[frame]
    return cropImagesAroundCMS(img, x, y, length, size)