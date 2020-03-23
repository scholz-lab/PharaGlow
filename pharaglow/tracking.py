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
    return filters.threshold_li(img, initial_guess = lambda arr: np.quantile(arr, 0.15))


@pims.pipeline
def preprocess(img, minSize = 800, threshold = None, smooth = 0, dilate = False):
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
    #mask = morphology.remove_small_holes(mask, area_threshold=12, connectivity=1, in_place=True)
    for i in range(dilate):
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


def calculateMask(frames, bgWindow = 15, thresholdWindow = 30, minSize = 50, subtract = False, smooth = 0, tfactor = 1, **kwargs):
    """standard median stack-projection to obtain a background image followd by thresholding and filtering of small objects to get a clean mask."""
    if subtract:
        bg = np.median(frames[::bgWindow], axis=0)
        #subtract bg from all frames
        frames = subtractBG(frames, bg)
    # image to determine threshold
    tmp = np.max(frames[::thresholdWindow], axis=0)
    # smooth
    if smooth:
        tmp = filters.gaussian(tmp, smooth, preserve_range = True)
    # get an overall threshold value and binarize images by using z-stack
    thresh = getThreshold(tmp)*tfactor
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
    if sx-xc>=length//2:
        warnings.warn('The object is larger than the bounding box. \
            Try increasing the length parameter.', Warning)
        img = img[:,xc - length//2:xc + length//2]
        mask = mask[:,xc - length//2:xc + length//2]
        xc = length//2
    if sy-yc>=length//2:
        warnings.warn('The object is larger than the bounding box. \
            Try increasing the length parameter.', Warning)
        img = img[yc - length//2:yc + length//2]
        mask = mask[yc - length//2:yc + length//2]
        yc = length//2
    sy, sx = img.shape
    yoff = length//2-yc
    xoff = length//2-xc
    if yoff<0 or xoff<0:
        warnings.warn('The center of mass is severly off center in this image. The image might be cropped.', \
                      Warning)
    if yoff>=0 and xoff>=0:
        im[yoff:yoff+sy, xoff:xoff+sx] = img*mask
    elif yoff<0 and xoff>=0:
        im[0:yoff+sy, xoff:sx+xoff] = (img*mask)[-yoff:]
    elif xoff<0 and yoff>=0:
        im[yoff:yoff+sy, :sx+xoff] = (img*mask)[:,-xoff:]
    else:
        im[0:yoff+sy, :sx+xoff] = (img*mask)[-yoff:,-xoff:]
    return im

def extractImagePad(img, bbox, pad, mask):
    # get a larger than bounding box image by padding some
    xmin, ymin, xmax, ymax  = region.bbox
    sliced = slice(np.max([0, xmin-pad]), xmax+pad), slice(np.max([0, ymin-pad]), ymax+pad)
    if mask:
        img[~mask] = 0
    return img[sliced]


def objectDetection(mask, img, params, frame, nextImg):
    """label binary image and extract a region of interest around the object."""
    df = pd.DataFrame()
    label_image = skimage.measure.label(mask, background=0, connectivity = 1)
    label_image = skimage.segmentation.clear_border(label_image, buffer_size=0, bgval=0, in_place=False, mask=None)
    diffImage = util.img_as_float(nextImg) - util.img_as_float(img)
    for region in skimage.measure.regionprops(label_image, intensity_image=img):
        if region.area > params['minSize'] and region.area < params['maxSize']:
            # get the image of an object
            #im = extractImage(region.intensity_image, region.image, params['length'], region.local_centroid)
            #diffIm = extractImage(diffImage[region.slice], region.image, params['length'], region.local_centroid)
            # go back to smaller images
            im = extractImagePad(img, region.slice, params['pad'], mask=None)
            diffIm = extractImagePad(diffImage, region.slice, params['pad'], mask=None)
            # Store features which survived to the criterions
            df = df.append([{'y': region.centroid[0],
                             'x': region.centroid[1],
                             'slice':region.slice,
                             'frame': frame,
                             'area': region.area,
                             'image': im.ravel(),
                             'yw': region.weighted_centroid[0],
                             'xw': region.weighted_centroid[1],
                             'diffI': diffIm.ravel(),
                             'shape': im.shape,
                             },])
        # do watershed to get crossing objects separated. 
        elif region.area > params['minSize']:
            image = mask[region.slice]
            labeled = refine(image, size = params['watershed'])
            for part in skimage.measure.regionprops(labeled, intensity_image=img[region.slice]):
                if part.area > params['minSize'] and part.area < 1.1*params['maxSize']:
                    # get the image of an object
                    #im = extractImage(part.intensity_image, part.image, params['length'], part.local_centroid)
                    #diffIm = extractImage(diffImage[part.slice], part.image, params['length'], part.local_centroid)
                    # go back to smaller images
                    im = extractImagePad(img, part.slice, params['pad'], mask=image)
                    diffIm = extractImagePad(diffImage, part.slice, params['pad'], mask=image)
                    # Store features which survived to the criterions
                    df = df.append([{'y': part.centroid[0],
                                     'x': part.centroid[1],
                                     'slice':part.slice,
                                     'frame': frame,
                                     'area': part.area,
                                     'image': im.ravel(),
                                     'yw': part.weighted_centroid[0],
                                     'xw': part.weighted_centroid[1],
                                     'diffI': diffIm.ravel(),
                                     'shape':im.shape,
                                     },])
    return df


def runfeatureDetection(frames, masks, params, frameOffset):
    """detect objects in each image and use region props to extract features and store a local image."""
    feat = []
    for num, img in enumerate(frames[:-1]):
        feat.append(objectDetection(masks[num], img, params, num+frameOffset, frames[num+1]))
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


def cropImagesAroundCMS(img, x, y, length, size, refine = False):
    """Using the interpolated center of mass coordindates (x,y), fill in missing images. img is a full size frame."""
    xmin, xmax = int(x - length//2), int(x + length//2)
    ymin, ymax = int(y-length//2), int(y+length//2)
    sliced = slice(np.max([0, ymin]), ymax), slice(np.max([0, xmin]), xmax)
    im = img[sliced]
    padx = [np.max([-xmin, 0]), np.max([xmax-img.shape[1], 0])]
    pady = [np.max([-ymin, 0]), np.max([ymax-img.shape[0], 0])]
    im = np.pad(im, [pady, padx] , mode='constant')
    # refine to a single animal if neccessary
    if refine:
        mask = preprocess(im, minSize = 0, threshold = None, smooth = 0)
        labeled = refine(mask, size)
        d = length//2
        if len(np.unique(labeled))>2:
            for part in skimage.measure.regionprops(labeled):
                d2 = np.sqrt((part.centroid[0]-length//2)**2+(part.centroid[1]-length//2)**2)
                if d2 < d:
                    mask = labeled==part.label
            im = im*mask
    return im.ravel()


def fillMissingImages(imgs, frame, x, y, length, size, refine = True):
    """run this on a dataframe to interpolate images from missing coordinates."""
    img = imgs[frame]
    return cropImagesAroundCMS(img, x, y, length, size, refine)


def fillMissingDifferenceImages(imgs, frame, x, y, length, size, refine = False):
    """run this on a dataframe to interpolate images from missing coordinates."""
    if frame<len(imgs):
        img = util.img_as_float(imgs[frame])-util.img_as_float(imgs[frame+1])
        return cropImagesAroundCMS(img, x, y, length, size)
    else:
        return np.zeros(length*length)