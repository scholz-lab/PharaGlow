#!/usr/bin/env python

"""util.py: useful general functions like filters and peak detection."""
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import math
import numpy as np
import pandas as pd


# def parallelize_dataframe(df, func, params, n_cores):

#     """ split a dataframe to easily use multiprocessing."""
#     df_split = np.array_split(df, n_cores)
#     df_split = [d for d in df_split  if len(d)>0]
#     if len(df_split) <1:
#         return
#     # filter zero-size jobs
#     print([len(d) for d in df_split])
#     pool = Pool(n_cores)
#     df = pd.concat(pool.starmap(func, zip(df_split, np.repeat(params, len(df_split)))))
#     pool.close()
#     pool.join()
#     return df


def parallel_analysis(args, param, parallelWorker, framenumbers = None,  nWorkers = 5, output= None, depth = 'uint8', **kwargs):
    """Use multiprocessing to speed up image analysis. This is inspired by the trackpy.batch function.

    Args:
        args (tuple): contains iterables eg. (frames, masks) or just frames that will be iterated through.
        param (dict): image analysis parameters
        parallelWorker (func): a function defining what should be done with args
        framenumbers (list, optional): a list of frame numbers corresponding to the frames in args. Defaults to None.
        nWorkers (int, optional): Processes to use, if 1 will run without multiprocessing. Defaults to 5.
        output ([type], optional): {None, trackpy.PandasHDFStore, SomeCustomClass} a storage class e.g. trackpy.PandasHDFStore. Defaults to None.
        depth (str, optional): bit depth of frames. Defaults to 'uint8'.

    Returns:
        output or (pandas.DataFrame, numpy.array)
    """
    if framenumbers is None:
        framenumbers = np.arange(len(args[0]))

    # Prepare wrapped function for mapping to `frames`
    detection_func = partial(parallelWorker, params = param)
    if nWorkers ==1:
        func = map
        pool = None
    else:
        # prepare imap pool
        pool = ProcessPoolExecutor(max_workers=nWorkers, mp_context=mp.get_context("spawn"))
        func = pool.map

    objects = []
    images = []
    try:
        for i, res in enumerate(func(detection_func, zip(*args, framenumbers))):
            if i%100 ==0:
                print(f'Analyzing image {i} of {len(args[0])}')
            if len(res[0]) > 0:
                # Store if features were found
                if output is None:
                    objects.append(res[0])
                    if len(res)>1:
                        images += res[1]
                else:
                    # here we keep images within the dataframe
                    if len(res)>1:
                        res[0]['images'] = res[1]
                    output.put(res[0])
    finally:
        if pool:
            # Ensure correct termination of Pool
            pool.shutdown()

    if output is None:
        if len(objects) > 0:
            objects = pd.concat(objects).reset_index(drop=True)
            if len(images)>0:
                images = np.array([pad_images(im, shape, param['length'], reshape = True, depth=depth) for im,shape in zip(images, objects['shapeX'])])
                images = np.array(images).astype(depth)
            return objects, images
        else:  # return empty DataFrame
            warnings.warn("No objects found in any frame.")
            return pd.DataFrame([]), images
    else:
        return output


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    import warnings
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2)][:len(x)]


def unravelImages(im, lengthX):
    """ Reshape images from linear to square.
    Args:
        im (list): an image
        lengthX (int): shape of new image in second axis

    Returns:
        numpy.array: image reshaped as (N,lengthX)
    """
    return im.reshape(-1, lengthX)


def get_im(df, colnames, lengthX):
    """deprecated
    """

    return unravelImages(df[colnames].to_numpy(), lengthX)


def pad_images(im, shape, size, reshape = True, depth = 'uint8'):
    """pad image to desired size.

    Args:
        im (list): a linearized version of an image
        shape (int): shape of second axis of image.
        size (int): pad image to size (size, size)
        reshape (bool, optional): reshape image to (-1, shape) before padding. Defaults to True.
        depth (str, optional): bit depth of image. Defaults to 'uint8'.

    Returns:
        numpy.array: padded image of size (size, size)
    """
    # make image from list
    im = np.array(im, dtype = depth)
    if reshape:
        im = im.reshape(-1, shape)
    # pad an image to square
    sy, sx = im.shape
    if sy > size or sx > size:
        warnings.warn(f'Image {sy, sx} larger than pad size {size}. Cropping')
        cropy, cropx = math.ceil(max([0, sy-size])/2), math.ceil(max([0, sx-size])/2)
        im = im[cropy:sy-cropy, cropx:sx-cropx]
        return pad_images(im, shape, size, reshape = False)
    # how much to add around each side
    py, px = (size-sy)//2, (size-sx)//2
    # add back the possible rounding error
    oy, ox = (size-sy)%2, (size-sx)%2
    new_im = np.pad(im, [(py, py+oy), (px, px+ox)], mode='constant', constant_values= 0)
    if new_im.shape !=(size, size):
        warnings.warn(f'Rerunning to correct size {size}')
        return pad_images(im, shape, size, reshape = False)
    return new_im
