#!/usr/bin/env python

"""util.py: useful general functions like filters and peak detection."""
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import gc
import math

def parallelize_dataframe(df, func, params, n_cores):
    """ split a dataframe to easily use multiprocessing."""
    df_split = np.array_split(df, n_cores)
    df_split = [d for d in df_split  if len(d)>0]
    if len(df_split) <1:
        return
    # filter zero-size jobs
    print([len(d) for d in df_split])
    pool = Pool(n_cores)
    df = pd.concat(pool.starmap(func, zip(df_split, np.repeat(params, len(df_split)))))
    pool.close()
    pool.join()
    return df


def parallel_analysis(args, param, parallelWorker, framenumbers = None,  nWorkers = 5, output= None):
    """use multiprocessing to speed up image analysis. This is inspired by the trackpy.batch function.
    arg:s contains iterables eg. (frames, masks) or just frames that will be iterated through.
    param: parameters given to all jobs
    parallelWorker: a function taking iterable args, and kwargs and returns a dataframe and one more result (optional)
    framenumbers: if given, these will replace the 'frames' columns. Default assumption is that frames are consecutive integers.
    nWorkers: processes to use, if 1 will run without multiprocessing
    process_results: a function to run on the resulting dataframe.

    output : {None, trackpy.PandasHDFStore, SomeCustomClass}
        If None, return all results as one big DataFrame. Otherwise, pass
        results from each frame, one at a time, to the put() method
        of whatever class is specified here.

    returns: specified output or the results as pd.DataFrame and any other result as list.
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
        pool = Pool(processes=nWorkers)
        func = pool.imap
        
    objects = []
    images = []
    try:
        for i, res in enumerate(func(detection_func, zip(*args, framenumbers))):
            if i%10 ==0:
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
            pool.terminate()

    if output is None:
        if len(objects) > 0:
            objects = pd.concat(objects).reset_index(drop=True)
            if len(images)>0:
                images = np.array([pad_images(im, shape, param['length'], reshape = True) for im,shape in zip(images, objects['shapeX'])])
                images = np.array(images).astype(np.uint8)
            return objects, images
        else:  # return empty DataFrame
            warnings.warn("No objects found in any frame.")
            return pd.DataFrame(columns=list(objects.columns) + ['frame']), images
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


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
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
    """reshape images from linear to square."""
    return im.reshape(-1, lengthX)
    
    
def get_im(df, colnames, lengthX):
    """get an image from a dataframe of columns for each pixel."""
    return unravelImages(df[colnames].to_numpy(), lengthX)


def pad_images(im, shape, size, reshape = True):
    # make image from list
    im = np.array(im, dtype = uint16)
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
    newIm = np.pad(im, [(py, py+oy), (px, px+ox)], mode='constant', constant_values= 0)
    if newIm.shape !=(size, size):
        warnings.warn(f'Rerunning to correct size {size}')
        return pad_images(im, shape, size, reshape = False)
    return newIm
