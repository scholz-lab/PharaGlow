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

import time
from multiprocessing import Pool
import gc

@pims.pipeline
def subtractBG(img, bg):
    """
    Subtract a background from the image.
    """
    tmp = img-bg
    mi, ma = np.min(tmp), np.max(tmp)
    tmp -= mi
    if ma != mi:
        tmp /=(ma - mi)
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
        print(num)
        if num ==2:
            return labelled
        if num<current_no and num>0:
            min_mask = labelled
            current_no = num
    return min_mask



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
    return img[sliced]


def objectDetection(mask, img, params, frame):
    """label binary image and extract a region of interest around the object."""
    df = pd.DataFrame()
    crop_images = pd.DataFrame()
    label_image = skimage.measure.label(mask, background=0, connectivity = 1)
    label_image = skimage.segmentation.clear_border(label_image, buffer_size=0, bgval=0, in_place=False, mask=None)
    #diffImage = util.img_as_float(nextImg) - util.img_as_float(img)
    for region in skimage.measure.regionprops(label_image, intensity_image=img):
        if region.area > params['minSize'] and region.area < params['maxSize']:
            # get the image of an object
            #im = extractImage(region.intensity_image, region.image, params['length'], region.local_centroid)
            #diffIm = extractImage(diffImage[region.slice], region.image, params['length'], region.local_centroid)
            # go back to smaller images
            im = extractImagePad(img, region.bbox, params['pad'], mask=label_image==region.label)
            #diffIm = extractImagePad(diffImage, region.bbox, params['pad'], mask=label_image==region.label)
            # bbox is min_row, min_col, max_row, max_col
            # Store features which survived to the criterions
            df = df.append([{'y': region.centroid[0],
                             'x': region.centroid[1],
                             'slice_x0':region.bbox[0],
                             'slice_x1':region.bbox[2],
                             'slice_y0':region.bbox[1],
                             'slice_y1':region.bbox[3],
                             'frame': frame,
                             'area': region.area,
                             #'image': im.ravel(),
                             'yw': region.weighted_centroid[0],
                             'xw': region.weighted_centroid[1],
                             'shapeY': im.shape[0],
                             'shapeX': im.shape[1],
                             },])
            # add the images to crop images
            crop_images = crop_images.append([list(im.ravel())])
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
                    im = extractImagePad(img, offsetbbox, params['pad'], mask=tmpMask)
                    #diffIm = extractImagePad(diffImage, offsetbbox, params['pad'], mask=tmpMask)
                    # Store features which survived to the criterions
                    df = df.append([{'y': part.centroid[0]+yo,
                                     'x': part.centroid[1]+xo,
                                     'slice_x0':offsetbbox[0],
                                     'slice_x1':offsetbbox[2],
                                     'slice_y0':offsetbbox[1],
                                     'slice_y1':offsetbbox[3],
                                     'frame': frame,
                                     'area': part.area,
                                     #'image': im.ravel(),
                                     'yw': part.weighted_centroid[0]+yo,
                                     'xw': part.weighted_centroid[1]+xo,
                                     'shapeY':im.shape[0],
                                     'shapeX': im.shape[1],
                                     },])
                    # add the images to crop images
                    crop_images = crop_images.append([list(im.ravel())])
    if not df.empty:
        df['shapeX'] = df['shapeX'].astype(int)
        df['shapeY'] = df['shapeY'].astype(int)
    
    return df, crop_images


def runfeatureDetection(frames, masks, params, frameOffset):
    """detect objects in each image and use region props to extract features and store a local image."""
    feat = []
    cropped_images = []
    print(f'Analyzing frames {frameOffset} to {frameOffset+len(frames)}')
    sys.stdout.flush()
    for num, img in enumerate(frames):
        df, crop_ims = objectDetection(masks[num], img, params, num+frameOffset)
        feat.append(df)
        cropped_images.append(crop_ims)
    features = pd.concat(feat)
    cropped_images = pd.concat(cropped_images)
    return features, cropped_images


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
    return traj.interpolate()


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


def fillMissingDifferenceImages(imgs, frame, x, y, lengthX, lengthY, size, refine = False):
    """run this on a dataframe to interpolate images from missing coordinates."""
    if frame<len(imgs):
        img = util.img_as_float(imgs[frame])-util.img_as_float(imgs[frame+1])
        im, *_  = cropImagesAroundCMS(img, x, y, lengthX, lengthY, size, refine)
        return im
    else:
        return np.zeros(lengthX*lengthY)


def parallelWorker(j):
    """deine a worker function for parallelization."""
    frames, masks, params, frameOffset = j
    return runfeatureDetection(frames, masks, params, frameOffset)
    
    
def pool_init():
    """deal with possible memory leak."""
    gc.collect()


def parallelDetection(rawframes, masks, param, nWorkers = 5, chunksize = 100):
    """utility function to run pharaglow as parallel using multiprocessing. 
    Chunks the images into smaller jobs and executes runfeatureDetection() for each job."""
    L = len(rawframes)
    # create chunks of analysis based on how many workers we use
    #slice the movie into pieces to run
    slices = zip((range(0,L, chunksize)), (range(chunksize,L+1, chunksize)))
    jobs = []
    for (a,b) in slices:
#         print(a,b)
        jobs.append([rawframes[a:b], masks[a:b], param, a])
    # add the remainder job for things not divisible by chunksize
    jobs.append([rawframes[b:], masks[b:], param, b])
    # delete jobs of length 1
    jobs = [j for j in jobs if len(j[0])>1]
    # add last bit
    #run the parallel feature detection.
    features = []#
    images = []
    gc.collect()
    p = Pool(processes = nWorkers, initializer=pool_init)
    start = time.time()
    for k, res in enumerate(p.imap_unordered(parallelWorker, jobs)):
        features.append(res[0])
        # replace nan values with 0 s and declare unit8 for images
        images.append(res[1].fillna(0).astype('uint8'))
        if k == nWorkers:
            print('Expected time is approx. {} s'.format((L/chunksize-k)*(time.time()-start)/nWorkers))
    
    p.close()
    p.join()
    features = pd.concat(features)
    images = pd.concat(images)
    # change the image column names
    images.columns = [f"im{s}" for s in images.columns]
    # make one big dataframe
    features = pd.concat([features, images], axis = 1)
    features = features.reset_index(drop=True)
    return features


def interpolate_helper(rawframes, row, param):
    """wrapper to make the code more readable. This interpolates all missing images in a trajectory.
    check if image is all zeros - then we insert an image from the original movie
    """
    im_cols =  [f'im{i}' for i in range(int(row['shapeX'])*int(row['shapeY']))]
    
    if np.sum(row[im_cols])==0:
        print('interpolating', row['frame'])
        im, sx0,sx1,sy0, sy1, ly, lx = fillMissingImages(rawframes, int(row['frame']), row['x'], row['y'],\
                                                   lengthX=row['shapeX'],lengthY=row['shapeY'], size=param['watershed'])
        # make the image into a pandas format and return a whole row
        im_cols =  [f'im{i}' for i in range(lx*ly)]
        # other column names
        [im_cols.append(x) for x in ['slice_x0','slice_x1','slice_y0','slice_y1', 'shapeY', 'shapeX']]
        row[im_cols] = [*list(im.ravel()),  sx0,sx1,sy0, sy1, ly, lx]
        
    return row
