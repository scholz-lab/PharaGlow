#!/usr/bin/env python

"""tracking.py: Detection of worms and trackpy-based worm tracking."""

import numpy as np
import pandas as pd
import warnings

import pims
import trackpy as tp

import skimage
from skimage.measure import label
from skimage import morphology, util, filters, segmentation, measure
from scipy import ndimage as ndi
from functools import partial
from multiprocessing import Pool
from scipy.stats import skew

from .util import pad_images

@pims.pipeline
def subtractBG(img, bg):
    """Subtract a background from the image.

    Args:
        img (numpy.array or pims.Frame): input image
        bg (numpy.array or pims.Frame): second image with background

    Returns:
        numpy.array: background subtracted image
    """
    tmp = img-bg
    mi, ma = np.min(tmp), np.max(tmp)
    tmp -= mi
    return util.img_as_float(tmp)


@pims.pipeline
def getThreshold(img):
    """"return a global threshold value for an image using yen's method.
    Returns:
        float: theshold value
    """
    return filters.threshold_yen(img)#, initial_guess = lambda arr: np.quantile(arr, 0.5))


@pims.pipeline
def preprocess(img, threshold = None, smooth = 0, dilate = False):
    """
    Apply image processing functions to return a binary image.

    Args:
        img (numpy.array or pims.Frame): input image
        smooth (int): apply a gaussian filter to img with width=smooth
        threshold (float): threshold value to apply after smoothing (default: None)
        dilate (int): apply a binary dilation n = dilate times (default = False)

    Returns:
        numpy.array: binary (masked) image
    """
    # smooth
    if smooth:
        img = filters.gaussian(img, smooth, preserve_range = True)
    # Apply thresholds
    if threshold ==None:
        threshold = filters.threshold_yen(img)
    mask = img >= threshold
    # dilations
    for i in range(dilate):
        mask = morphology.dilation(mask)
    return mask


@pims.pipeline
def refineWatershed(img, min_size, filter_sizes = [3,4,5], dilate = 0):
    """"Refine segmentation using thresholding with different filtered images.
    Favors detection of two objects.
    Args:
        img (numpy.array or pims.Frame): input image
        min_size (int, float): minimal size of objects to retain as labels
        filter_sizes (list, optional): filter sizes to try until objects are separated. Defaults to [3,4,5].
        dilate (int, optional): dilate as often as in the original mask to keep sizes consistent
    Returns:
        numpy.array : labelled image
    """
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
        #NOTE: in_place argument is deprecated. The default behavior is creating a new array.
        #REF: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects
        #mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=2, in_place=True)
        mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=2)
        labelled, num = label(mask, background=0, connectivity = 2,return_num=True)
        if num ==2:
            min_mask = labelled
        if num<current_no and num>0:
            min_mask = labelled
            current_no = num
    min_mask = min_mask.astype(int)
    # dilations
    for i in range(dilate):
        min_mask = morphology.dilation(min_mask)
    return min_mask


def calculateMask(frames, bgWindow = 30 , thresholdWindow = 30, subtract = False, smooth = 0, tfactor = 1, **kwargs):
    """standard median stack-projection to obtain a background image followd by
    thresholding and filtering of small objects to get a clean mask.

    Args:
        frames (numpy.array or pims.ImageSequence): image stack with input images
        bgWindow (int): subsample frames for background creation by selecting bgWindow numbers of frames evenly spaced. Defaults to 30.
        thresholdWindow (int, optional): subsample frames to calculate the threshold.
                        Selects thresholdWindow evenly spread frames. Defaults to 30.
        subtract (bool, optional): calculate and subtract a median-background. Defaults to False.
        smooth (int, optional): size of gaussian filter for image smoothing. Defaults to 0.
        tfactor (int, optional): fudge factor to correct threshold. Discouraged. Defaults to 1.

    Returns:
        numpy.array: masked (binary) image array
    """

    if subtract:
        select_frames = np.linspace(0, len(frames)-1, bgWindow).astype(int)
        bg = np.median(frames[select_frames], axis=0)
        if np.max(bg) > 0:
            #subtract bg from all frames
            frames = subtractBG(frames, bg)
    # image to determine threshold
    select_frames = np.linspace(0, len(frames)-1, thresholdWindow).astype(int)
    tmp = np.max(frames[select_frames], axis=0)
    # smooth
    if smooth:
        tmp = filters.gaussian(tmp, smooth, preserve_range = True)
    # get an overall threshold value and binarize images by using z-stack
    thresh = getThreshold(tmp)*tfactor
    return preprocess(frames, threshold = thresh, **kwargs)


def extractImage(img, mask, length, cmsLocal):
    """ extracts a square image of an object centered around center of mass coordinates with size (length, length). Mask ensures that
    only one object is visible if two are in the same region.
    img is the bounding box region of the object.

    Args:
        img (numpy.array or pims.Frame): larger image
        mask (numpy.array): binary mask of the same size as img
        length (int): length of resulting image
        cmsLocal (float, float): center point

    Returns:
        numpy.array: square cutout of (length,length) will be returned
    """

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
    """get a larger than bounding box image by padding around the detected object.

    Args:
        img (numpy.array): input image
        bbox (tuple): bounding box which lies in img in format (ymin, xmin, ymax, xmax)
        pad (int): number of pixels to pad around each size. reslting image will be larger by 2*pad on each side.
        mask (numpy.array, optional): binary mask of size img. Defaults to None.

    Returns:
        numpy.array: padded image
        slice: location/bbox of padded image in original image
    """

    ymin, xmin, ymax, xmax  = bbox
    sliced = slice(np.max([0, ymin-pad]), ymax+pad), slice(np.max([0, xmin-pad]), xmax+pad)
    if mask is not None:
        assert mask.shape == img.shape
        img = img*mask
    return img[sliced], sliced


def objectDetection(mask, img, frame, params):
    """label a binary image and extract a region of interest around each labelled object,
        as well as collect properties of the object in a DataFrame.

    Args:
        mask (numpy.array): binary image
        img (numpy.array): intensity image with same shape as mask
        frame (int): a number to indicate a time stamp, which will populate the column 'frame'
        params (dict): parameter dictionary containing image analysis parameters.

    Returns:
        pandas.Dataframe, list: dataframe with information for each image, list of corresponding images.
    """
    assert mask.shape == img.shape, 'Image and Mask size do not match.'
    #NOTE: appending lists, then creating DataFrame REF: https://stackoverflow.com/a/75956237
    y_list = []
    x_list = []
    slice_y0_list = []
    slice_y1_list = []
    slice_x0_list = []
    slice_x1_list = []
    frame_list = []
    area_list = []
    yw_list = []
    xw_list = []
    shapeY_list = []
    shapeX_list = []
    crop_images = []
    label_image = measure.label(mask, background=0, connectivity = 1)
    #label_image = segmentation.clear_border(label_image, buffer_size=0, bgval=0, in_place=False, mask=None)
    #NOTE: in_place argument is deprecated. The default behavior is creating a new array.
    #REF: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.clear_border
    label_image = segmentation.clear_border(label_image, buffer_size=0, bgval=0, mask=None)

    for region in measure.regionprops(label_image, intensity_image=img):
        if region.area > params['minSize'] and region.area < params['maxSize']:
            # get the image of an object
            im, sliced = extractImagePad(img, region.bbox, params['pad'], mask=label_image==region.label)
            bbox = [sliced[0].start, sliced[1].start, sliced[0].stop, sliced[1].stop]
            # bbox is min_row, min_col, max_row, max_col
            # Store features which survived to the criterions
            y_list.append(region.centroid[0])
            x_list.append(region.centroid[1])
            slice_y0_list.append(bbox[0])
            slice_y1_list.append(bbox[2])
            slice_x0_list.append(bbox[1])
            slice_x1_list.append(bbox[3])
            frame_list.append(frame)
            area_list.append(region.area)
            yw_list.append(region.weighted_centroid[0])
            xw_list.append(region.weighted_centroid[1])
            shapeY_list.append(im.shape[0])
            shapeX_list.append(im.shape[1])
            # add the images to crop images
            crop_images.append(list(im.ravel()))
        # do watershed to get crossing objects separated.
        elif region.area > params['minSize']:
            labeled = refineWatershed(img[region.slice], min_size = params['watershed'], dilate = params['dilate'])
            for part in measure.regionprops(labeled, intensity_image=img[region.slice]):
                if part.area > params['minSize']*0.75 and part.area < params['maxSize']:
                    # get the image of an object
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
                    y_list.append(part.centroid[0]+yo)
                    x_list.append(part.centroid[1]+xo)
                    slice_y0_list.append(bbox[0])
                    slice_y1_list.append(bbox[2])
                    slice_x0_list.append(bbox[1])
                    slice_x1_list.append(bbox[3])
                    frame_list.append(frame)
                    area_list.append(part.area)
                    yw_list.append(part.weighted_centroid[0]+yo)
                    xw_list.append(part.weighted_centroid[1]+xo)
                    shapeY_list.append(im.shape[0])
                    shapeX_list.append(im.shape[1])
                    # add the images to crop images
                    crop_images.append(list(im.ravel()))

    info_images = {
        'y': y_list,
        'x': x_list,
        'slice_y0': slice_y0_list,
        'slice_y1': slice_y1_list,
        'slice_x0': slice_x0_list,
        'slice_x1': slice_x1_list,
        'frame': frame_list,
        'area': area_list,
        'yw': yw_list,
        'xw': xw_list,
        'shapeY': shapeY_list,
        'shapeX': shapeX_list
    }

    df = pd.DataFrame(info_images)
    if not df.empty:
        df['shapeX'] = df['shapeX'].astype(int)
        df['shapeY'] = df['shapeY'].astype(int)

    return df, crop_images


def linkParticles(df, searchRange, minimalDuration, **kwargs):
    """ Link detected particles into trajectories.
    **kwargs can be passed to the trackpy function link_df to modify tracking behavior.

    Args:
        df (pandas.DataFrame): pandas dataframe that contains at least the columns 'frame' and 'x', 'y'.
        searchRange (float): how far particles can move in one frame
        minimalDuration (int): minimal duration of a track in frames

    Returns:
        pandas.DataFrame: inplace modified dataframe with an added column called 'particles' which labels the objects belonging to one trajectory.
    """

    traj = tp.link_df(df, searchRange, **kwargs)
    # filter short trajectories
    traj = tp.filter_stubs(traj, minimalDuration)
    # make a numerical index
    traj.set_index(np.arange(len(traj.index)), inplace = True)
    return traj


def interpolateTrajectories(traj, columns = None):
    """given a dataframe with a trajectory, interpolate missing frames.
    The interpolate function ignores non-pandas types, so some columns will not be interpolated.

    Args:
        traj (pandas.DataFrame): pandas dataframe containing at minimum the columns 'frame' and the columns given in colums.
        columns (list(str), optional): list of columns to interpolate.
        Defaults to None, which means all columns are attempted to be interpolated.

    Returns:
        pandas.DataFrame: dataframe with interpolated trajectories
    """

    idx = pd.Index(np.arange(traj['frame'].min(), traj['frame'].max()+1), name="frame")
    traj = traj.set_index("frame").reindex(idx).reset_index()
    if columns is not None:
        for c in columns:
            traj[c] = traj[c].interpolate()
        return traj
    return traj.interpolate(axis = 0)


def cropImagesAroundCMS(img, x, y, lengthX, lengthY, size, refine = False):
    """Using the interpolated center of mass coordindates (x,y), fill in missing images. img is a full size frame.

    Args:
        img (numpy.array): original image
        x (float): x-coordinate
        y (float): y-coordinate
        lengthX (int): length of resulting image
        lengthY (int): length of resulting image
        size (float): expected minimal size for a relevant object
        refine (bool, optional): Use filtering to separate potentially colliding objects. Defaults to False.

    Returns:
        list: image unraveled as 1d list
        tuple: bounding box
        int: length of first image axis
        int: length of second image axis
    """

    xmin, xmax = int(x - lengthX//2), int(x + lengthX//2)
    ymin, ymax = int(y-lengthY//2), int(y+lengthY//2)
    sliced = slice(np.max([0, ymin]), np.min(ymax)), slice(np.max([0, xmin]), xmax)
    im = img[sliced]
    # actual size in case we went out of bounds
    ly, lx = im.shape
    # refine to a single animal if neccessary
    if refine:
        labeled = refineWatershed(im, size)
        d = np.sqrt(lx**2+ly**2)
        if len(np.unique(labeled))>2:
            for part in measure.regionprops(labeled):
                d2 = np.sqrt((part.centroid[0]-ly//2)**2+(part.centroid[1]-lx//2)**2)
                if d2 < d:
                    mask = labeled==part.label
                    d = d2
            im = im*mask
    # make bounding box from slice. Bounding box is [ymin, xmin, ymax, xmax]
    bbox = [sliced[0].start, sliced[1].start, sliced[0].stop, sliced[1].stop]
    return im.ravel(), bbox, ly, lx


def fillMissingImages(imgs, frame, x, y, lengthX, lengthY, size, refine = False):
    """ Run this on a dataframe to interpolate images from previously missing, now interpolated coordinates.

    Args:
        img (numpy.array): original image
        x (float): x-coordinate
        y (float): y-coordinate
        lengthX (int): length of resulting image
        lengthY (int): length of resulting image
        size (float): expected minimal size for a relevant object
        refine (bool, optional): Use filtering to separate potentially colliding objects. Defaults to False.

    Returns:
        list: image unraveled as 1d list
        int: ymin of bounding box
        int: xmin of bounding box
        int: ymax of bounding box
        int: xmax of bounding box
        int: length of first image axis
        int: length of second image axis
    """
    img = imgs[frame]
    im, sliced, ly, lx = cropImagesAroundCMS(img, x, y, lengthX, lengthY, size, refine)
    return im, sliced[0],sliced[1],sliced[2],sliced[3], ly, lx


def parallelWorker(args, **kwargs):
    """helper wrapper to run object detection with multiprocessing.

    Args:
        args (div.): arguments for .tracking.objectDetection

    Returns:
        pandas.DataFrame: dataframe with information for each image
        list: list of corresponding images.
    """
    return objectDetection(*args, **kwargs)


def parallel_imageanalysis(frames, masks, param, framenumbers = None, parallelWorker= parallelWorker, nWorkers = 5, output= None):
    """use multiptocessing to speed up image analysis. This is inspired by the trackpy.batch function.

    frames: numpy.array or other iterable of images
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
            return pd.DataFrame(columns=list(objects.columns) + ['frame']), images
    else:
        return output


def interpolate_helper(rawframes, ims, tmp, param, columns = ['x', 'y', 'shapeX', 'shapeY', 'particle']):
    """wrapper to make the code more readable. This interpolates all missing images in a trajectory.
    check if currently the image is all zeros - then we insert an small image from the original movie around the interpolated coordinates.

    Args:
        rawframes (pims.ImageSequence): sequence of images
        ims (numpy.array): stack of small images around detected objects corresponding to rows in tmp
        tmp (pandas.DataFrame): pandas dataframe with an onject and its properties per row
        param (dict): dictionary of image analysis parameters, see example file `AnalysisParameters_1x.json`
        columns (list, optional): columns to interpolate. Defaults to ['x', 'y', 'shapeX', 'shapeY', 'particle'].

    Returns:
        pandas.DataFrame: interpolated version of tmp with missing values interpolated
        numpy.array: array of images with interpolated images inserted at the appropriate indices
    """

    # create a new column keeping track if this row is interpolated or already in the image stack
    tmp.insert(0, 'has_image', 1)
    tmp.insert(0, 'image_index', np.arange(len(ims)))
    # generate an interpolated trajectory where all frames are accounted for
    traj_interp = interpolateTrajectories(tmp, columns = columns)
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
