#!/usr/bin/env python

"""tracking.py: trackpy based worm tracking."""

import numpy as np
import pandas as pd
import warnings

import pims
import trackpy as tp
import os
import sys
import skimage
from skimage.measure import label
# preprocessing the image for tracking. 
from skimage import morphology, util, filters, segmentation
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max, canny
from functools import partial
import time
from multiprocessing import Pool
import gc
from .util import pad_images 

@pims.pipeline
def subtractBG(img, bg):
    """
    Subtract a background from the image.
    """
    tmp = img-bg
    mi, ma = np.min(tmp), np.max(tmp)
    tmp -= mi
    #if ma != mi:
    #    tmp /=(ma - mi)
    return util.img_as_float(tmp)


@pims.pipeline
def getThreshold(img):
    """"return a global threshold value"""
    return filters.threshold_yen(img)#, initial_guess = lambda arr: np.quantile(arr, 0.5))


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
def refineWatershed(img, size, filter_sizes = [3,4,5]):
    """Refine segmentation using refined thresholding."""
    min_mask = np.zeros(img.shape)
    current_no = np.inf
    for s in filter_sizes:
        bg = filters.gaussian(img, s, preserve_range = True)
        img = filters.gaussian(img-bg, 1)
        img[img<0] = 0
        img = img.astype(int)
        # mask 
        mask = img>filters.threshold_li(img, initial_guess = np.min)
        mask = ndi.binary_closing(mask)
        mask = morphology.remove_small_objects(mask, min_size=size, connectivity=2, in_place=True)
        labelled, num = label(mask, background=0, connectivity = 2,return_num=True)
        if num ==2:
            return labelled
        if num<current_no and num>0:
            min_mask = labelled
            current_no = num
    return min_mask.astype(int)



def calculateMask(frames, bgWindow = 15, thresholdWindow = 30, minSize = 50, subtract = False, smooth = 0, tfactor = 1, **kwargs):
    """standard median stack-projection to obtain a background image followd by thresholding and filtering of small objects to get a clean mask."""
    if subtract:
        bg = np.median(frames[::bgWindow], axis=0)
        if np.max(bg) > 0:
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

def extractImagePad(img, bbox, pad, mask=None):
    """ get a larger than bounding box image by padding around the detected object.
        returns two pandas dataframes:
        one with each row containing a detected object, its (x,y) location in px and additional properties.
        The second with unraveled images of each object.
    """
    ymin, xmin, ymax, xmax  = bbox
    sliced = slice(np.max([0, ymin-pad]), ymax+pad), slice(np.max([0, xmin-pad]), xmax+pad)
    if mask is not None:
        img = img*mask
    return img[sliced], sliced


def objectDetection(mask, img, frame, params):
    """label binary image and extract a region of interest around the object."""
    df = pd.DataFrame()
    crop_images = []
    label_image = skimage.measure.label(mask, background=0, connectivity = 1)
    label_image = skimage.segmentation.clear_border(label_image, buffer_size=0, bgval=0, in_place=False, mask=None)
    #diffImage = util.img_as_float(nextImg) - util.img_as_float(img)
    for region in skimage.measure.regionprops(label_image, intensity_image=img):
        if region.area > params['minSize'] and region.area < params['maxSize']:
            # get the image of an object
            #im = extractImage(region.intensity_image, region.image, params['length'], region.local_centroid)
            #diffIm = extractImage(diffImage[region.slice], region.image, params['length'], region.local_centroid)
            # go back to smaller images
            im, sliced = extractImagePad(img, region.bbox, params['pad'], mask=label_image==region.label)
            bbox = [sliced[0].start, sliced[1].start, sliced[0].stop, sliced[1].stop]
            #diffIm = extractImagePad(diffImage, region.bbox, params['pad'], mask=label_image==region.label)
            # bbox is min_row, min_col, max_row, max_col
            # Store features which survived to the criterions
            df = df.append([{'y': region.centroid[0],
                             'x': region.centroid[1],
                             'slice_y0':bbox[0],
                             'slice_y1':bbox[2],
                             'slice_x0':bbox[1],
                             'slice_x1':bbox[3],
                             'frame': frame,
                             'area': region.area,
                             #'image': im.ravel(),
                             'yw': region.weighted_centroid[0],
                             'xw': region.weighted_centroid[1],
                             'shapeY': im.shape[0],
                             'shapeX': im.shape[1],
                             },])
            # add the images to crop images
            crop_images.append(list(im.ravel()))
        # do watershed to get crossing objects separated. 
        elif region.area > params['minSize']:
            labeled = refineWatershed(img[region.slice], size = params['watershed'])
            for part in skimage.measure.regionprops(labeled, intensity_image=img[region.slice]):
                if part.area > params['minSize']*0.75 and part.area < params['maxSize']:
                    # get the image of an object
                    #im = extractImage(part.intensity_image, part.image, params['length'], part.local_centroid)
                    #diffIm = extractImage(diffImage[part.slice], part.image, params['length'], part.local_centroid)
                    # account for the offset from the region
                    yo, xo,_,_ = region.bbox
                    offsetbbox = np.array((part.bbox))+np.array([yo,xo,yo,xo])
                    # go back to smaller images
                    tmpMask = np.zeros(img.shape)
                    tmpMask[region.slice] = labeled==part.label
                    tmpMask = tmpMask.astype(int)
                    im, sliced = extractImagePad(img, offsetbbox, params['pad'], mask=tmpMask)
                    bbox = [sliced[0].start, sliced[1].start, sliced[0].stop, sliced[1].stop]
                    #diffIm = extractImagePad(diffImage, offsetbbox, params['pad'], mask=tmpMask)
                    # Store features which survived to the criterions
                    df = df.append([{'y': part.centroid[0]+yo,
                                     'x': part.centroid[1]+xo,
                                     'slice_y0':bbox[0],
                                     'slice_y1':bbox[2],
                                     'slice_x0':bbox[1],
                                     'slice_x1':bbox[3],
                                     'frame': frame,
                                     'area': part.area,
                                     #'image': im.ravel(),
                                     'yw': part.weighted_centroid[0]+yo,
                                     'xw': part.weighted_centroid[1]+xo,
                                     'shapeY':im.shape[0],
                                     'shapeX': im.shape[1],
                                     },])
                    # add the images to crop images
                    crop_images.append(list(im.ravel()))
    if not df.empty:
        df['shapeX'] = df['shapeX'].astype(int)
        df['shapeY'] = df['shapeY'].astype(int)
    
    return df, crop_images


# def runfeatureDetection(frames, masks, frameIndex, params):
#     """detect objects in each image and use region props to extract features and store a local image.
#         frames: contains the image stack
#         masks: matching binary stack
#         frameIndex: indicates a time corresponding to each image e.g. integers or timestamps.
#         params: analysis parameters passed to objectDetection()
#     """
#     feat = []
#     cropped_images = []
#     print(f'Analyzing frames {frameIndex[0]} to {frameIndex[-1]}')
#     sys.stdout.flush()
#     for num, img in enumerate(frames):
#         df, crop_ims = objectDetection(masks[num], img, frameIndex[num], params)
#         feat.append(df)
#         cropped_images+= crop_ims
#     features = pd.concat(feat)
#     features = features.reset_index()
#     return features, cropped_images


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


def interpolateTrajectories(traj, columns = None):
    """given a dataframe with a trajectory, interpolate missing frames.
    The interpolate function ignores non-pandas types, so no image interpolations."""
    idx = pd.Index(np.arange(traj['frame'].min(), traj['frame'].max()+1), name="frame")
    traj = traj.set_index("frame").reindex(idx).reset_index()
    if columns is not None:
        for c in columns:
            traj[c] = traj[c].interpolate()
        return traj
    return traj.interpolate(axis = 1)


def cropImagesAroundCMS(img, x, y, lengthX, lengthY, size, refine = False):
    """Using the interpolated center of mass coordindates (x,y), fill in missing images. img is a full size frame."""
    xmin, xmax = int(x - lengthX//2), int(x + lengthX//2)
    ymin, ymax = int(y-lengthY//2), int(y+lengthY//2)
    sliced = slice(np.max([0, ymin]), np.min(ymax)), slice(np.max([0, xmin]), xmax)
    im = img[sliced]
    # actual size in case we went out of bounds
    ly, lx = im.shape
    #padx = [np.max([-xmin, 0]), np.max([xmax-img.shape[1], 0])]
    #pady = [np.max([-ymin, 0]), np.max([ymax-img.shape[0], 0])]
    #im = np.pad(im, [pady, padx] , mode='constant')
    # refine to a single animal if neccessary
    if refine:
    #    #mask = preprocess(im, minSize = 0, threshold = None, smooth = 0)
        labeled = refineWatershed(im, size)
        d = np.sqrt(lx**2+ly**2)
        if len(np.unique(labeled))>2:
            for part in skimage.measure.regionprops(labeled):
                d2 = np.sqrt((part.centroid[0]-ly//2)**2+(part.centroid[1]-lx//2)**2)
                if d2 < d:
                    mask = labeled==part.label
                    d = d2
            im = im*mask
    # make bounding box from slice. Bounding box is [ymin, xmin, ymax, xmax]
    bbox = [sliced[0].start, sliced[1].start, sliced[0].stop, sliced[1].stop]
    return im.ravel(), bbox, ly, lx


def fillMissingImages(imgs, frame, x, y, lengthX, lengthY, size, refine = False):
    """run this on a dataframe to interpolate images from missing coordinates."""
    img = imgs[frame]
    im, sliced, ly, lx = cropImagesAroundCMS(img, x, y, lengthX, lengthY, size, refine)
    return im, sliced[0],sliced[1],sliced[2],sliced[3], ly, lx



def parallelWorker(args, **kwargs):
    return objectDetection(*args, **kwargs)


def parallel_imageanalysis(frames, masks, param, framenumbers = None, parallelWorker= parallelWorker, nWorkers = 5, output= None):
    """use multiptocessing to speed up image analysis. This is inspired by the trackpy.batch function.
    
    frames: array or other iterable of images
    masks: the binary of the frames, same length
    param: parameters given to all jobs
    
    output : {None, trackpy.PandasHDFStore, SomeCustomClass}
        If None, return all results as one big DataFrame. Otherwise, pass
        results from each frame, one at a time, to the put() method
        of whatever class is specified here.
    """
    assert len(frames) == len(masks), "unequal length of images and binary masks."
    if framenumbers is None:
        framenumbers = np.arange(len(frames))
    # Prepare wrapped function for mapping to `frames`
    
    detection_func = partial(parallelWorker, params = param)
    if nWorkers ==1:
        func = map
        pool = None
    else:
        # prepare imap pool
        pool = Pool(processes=nWorkers)
        func = pool.imap
        
    objects = []
    images = []
    try:
        for i, res in enumerate(func(detection_func, zip( masks,frames, framenumbers))):
            # allow alternate frame numbers
            if len(res[0]) > 0:
                # Store if features were found
                if output is None:
                    objects.append(res[0])
                    images += res[1]
                else:
                    # here we keep images within the dataframe
                    res[0]['images'] = res[1]
                    output.put(res[0])
    finally:
        if pool:
            # Ensure correct termination of Pool
            pool.terminate()
    
    if output is None:
        
        if len(objects) > 0:
            objects = pd.concat(objects).reset_index(drop=True)
            images = np.array([pad_images(im, shape, param['length']) for im,shape in zip(images, objects['shapeX'])])
            images = np.array(images).astype(np.uint8)
            return objects, images
        else:  # return empty DataFrame
            warnings.warn("No objects found in any frame.")
            return pd.DataFrame(columns=list(features.columns) + ['frame']), images
    else:
        return output


def interpolate_helper(rawframes, ims, tmp, param):
    """wrapper to make the code more readable. This interpolates all missing images in a trajectory.
    check if image is all zeros - then we insert an small image from the original movie around the interpolated coordinates.
    """
    # create a new column keeping track if this row is interpolated or already in the image stack
    tmp.insert(0, 'has_image', 1)
    tmp.insert(0, 'image_index', np.arange(len(ims)))
    # generate an interpolated trajectory where all frames are accounted for
    traj_interp = interpolateTrajectories(tmp, columns = ['x', 'y', 'shapeX', 'shapeY', 'particle'])
    # make sure we have a range index
    traj_interp.reset_index()
    # iterate through the dataframe and if the image is all nan, attempt to fill it
    images = []
    for idx, row in traj_interp.iterrows():
        if np.isnan(row['has_image']):
            # get the image
            im, sy0, sx0, sy1, sx1, ly, lx = fillMissingImages(rawframes, int(row['frame']), row['x'], row['y'],\
                                                   lengthX=row['shapeX'],lengthY=row['shapeY'], size=param['watershed'])
            # pad the image
            im = pad_images(im, lx, param['length'])
            # insert it into the array at the correct position
            images.append(im)
            # update the slice and shape information
            cols = ['slice_y0','slice_x0','slice_y1','slice_x1', 'shapeY', 'shapeX']
            traj_interp.loc[idx, cols] = sy0, sx0, sy1, sx1, ly, lx
        else:
            images.append(ims[int(row['image_index'])])
    return traj_interp, np.array(images)
