#!/usr/bin/env python

"""features.py: image analysis of pharynx. Uses skimage to provide image functionality."""
import pims
import numpy as np
from numpy.linalg import norm
from scipy.stats import skew
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize, disk, remove_small_holes, remove_small_objects, binary_closing, binary_opening
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from skimage.transform import rescale
from skimage.filters import rank, threshold_otsu, threshold_yen, gaussian
from skimage.measure import find_contours, profile_line, regionprops, label


def find_lawn(image, smooth = 1, area_holes = 15000, area_spots = 50000):
    """binarize the image of the bacterial lawn.

    Args:
        image (np.array or pims.Frame): image of a bacterial lawn
        smooth (int, optional): apply gaussian filter of size smooth px. Defaults to 1.
        area_holes (int, optional): remove small holes in binary image. Defaults to 15000.
        area_spots (int, optional): remove small objects in binary image. Defaults to 50000.

    Returns:
        np.array: binarized image
    """

    image = gaussian(image, smooth, preserve_range = True)
    thresh = threshold_otsu(image)
    binary = image > thresh
    binary = remove_small_holes(binary, area_threshold=area_holes, connectivity=1, in_place=False)
    binary = remove_small_objects(binary, min_size=area_spots, connectivity=8, in_place=False)
    return binary

@pims.pipeline
def thresholdPharynx(img):
    """Use Yen threshold to obtain mask of pharynx.

    Args:
        im (numpy.array or pims.Frame): image

    Returns:
        np.array: binary image with only the largest object
    """

    mask = img>threshold_yen(img)
    mask = binary_opening(mask)
    mask = binary_closing(mask)
    labeled = label(mask)
    # keep only the largest item
    area = 0
    for region in regionprops(labeled):
        if area <= region.area:
            mask = labeled==region.label
            area = region.area
    return mask


def skeletonPharynx(mask):
    """ Use skeletonization to obatain midline of pharynx.

    Args:
        mask (numpy.array): binary mask of the pharynx

    Returns:
        numpy.array: skeleton of mask
    """

    return skeletonize(mask)

@pims.pipeline
def sortSkeleton(skeleton):
    """Use hierarchical clustering with optimal ordering to get \
        the best path through the skeleton points.

    Args:
        skeleton (numpy.array): skeletonized image of an object

    Returns:
        list: list of coordinates ordered by distance
    """

    # coordinates of skeleton
    ptsX, ptsY = np.where(skeleton)
    # cluster
    Z = linkage(np.c_[ptsX, ptsY], method='average', metric='cityblock', optimal_ordering=True)
    return leaves_list(Z)


def pharynxFunc(x, *p, deriv = 0):
    """ Defines a cubic polynomial helper function.

    Args:
        x (numpy.array or list): list of coordinates to evaluate function on
        deriv (int, optional): return the polynomial or its first derivative Defaults to 0. {0,1}

    Returns:
        numpy.array or list: polynomial evaluated at x
    """

    if deriv==1:
        return p[1] + 2*p[2]*x + 3*p[3]*x**2 #+ 4*p[4]*x**4
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 #+ p[4]*x**5


def fitSkeleton(ptsX, ptsY, func = pharynxFunc):
    """Fit a (cubic) polynomial spline to the centerline. The input should be sorted skeleton coordinates.

    Args:
        ptsX (numpy.array or list): sorted x coordinates
        ptsY (numpy.array or list): sorted y coordinates
        func (function, optional): function to fit. Defaults to pharynxFunc.

    Returns:
        array: optimal fit parameters of pharynxFunc
    """

    nP = len(ptsX)
    x = np.linspace(0, 100, nP)
    # fit each axis separately
    poptX, _ = curve_fit(func, x, ptsX, p0=(np.mean(ptsX),1,1,0.1))
    poptY, _= curve_fit(func, x, ptsY, p0 = (np.mean(ptsY),1,1,0.1))

    return poptX, poptY


def morphologicalPharynxContour(mask, scale = 4, **kwargs):
    """ Uses morphological contour finding on a mask image to get a nice outline.
        We will upsample the image to get sub-pixel outlines.
        **kwargs are handed to morphological_chan_vese.

    Args:
        mask (numpy.array):  binary mask of pharynx.
        scale (int, optional): Scale to upsample the image by. Defaults to 4.

    Returns:
        numpy.array: coordinates of the contour as array of (N,2) coordinates.
    """

    # upscale this image to get accurate contour
    image = img_as_float(rescale(mask, scale))
    # intialize a checkerboard
    init_ls = checkerboard_level_set(image.shape, 5)
    # run morphological contour finding
    snake =  morphological_chan_vese(image, 10, init_level_set=init_ls, **kwargs)
    # let's try the contour
    contour= find_contours(snake, level = 0.5)#, fully_connected='high', positive_orientation='high',)
    # just in case we find multiple, get only the longest contour
    contour = contour[np.argmax([len(x) for x in contour])]
    cX, cY = np.array(contour/scale).T
    contour = np.stack((cX, cY), axis =1)
    return contour


def cropcenterline(poptX, poptY, contour):
    """ Define start and end point of centerline by crossing of contour.

    Args:
        poptX (array): optimal fit parameters describing pharynx centerline.
        poptY (array): optimal fit parameters describing pharynx centerline.
        contour (numpy.array): (N,2) array of points describing the pharynx outline.

    Returns:
        float, float: start and end coordinate to apply to .features._pharynxFunc(x) to create a centerline
    spanning the length of the pharynx.
    """

    xs = np.linspace(-50,150, 200)
    tmpcl = np.c_[pharynxFunc(xs, *poptX), pharynxFunc(xs, *poptY)]
    # update centerline based on crossing the contour
    # we are looking for two crossing points
    distClC = np.sum((tmpcl-contour[:,np.newaxis])**2, axis =-1)
    start, end = np.argsort(np.min(distClC, axis = 0))[:2]
     # update centerline length
    xstart, xend = xs[start],xs[end]
    # check if length makes sense, otherwise retain original
    if np.abs(start-end) < 50:
        xstart, xend = 0, 100

    return xstart, xend


def centerline(poptX, poptY, xs):
    """create a centerline from fitted function.

    Args:
        poptX (array): optimal fit parameters describing pharynx centerline.
        poptY (array): optimal fit parameters describing pharynx centerline.
        xs (np.array): array of coordinates to create centerline from .feature._pharynxFunc(x, *p, deriv = 0)

    Returns:
        numpy.array: (N,2) a centerline spanning the length of the pharynx. Same length as xs.
    """

    return np.c_[pharynxFunc(xs, *poptX), pharynxFunc(xs, *poptY)]


def normalVecCl(poptX, poptY, xs):
    """ Create vectors normal to the centerline by using the derivative of the function describing the midline.

    Args:
        poptX (array): optimal fit parameters describing pharynx centerline.
        poptY (array): optimal fit parameters describing pharynx centerline.
        xs (np.array): array of coordinates to create centerline from .feature._pharynxFunc(x, *p, deriv = 0)

    Returns:
        numpy.array: : (N,2) array of unit vectors orthogonal to centerline. Same length as xs.
    """

    # make an orthogonal vector to the cl by calculating derivative (dx, dy) and using (-dy, dx) as orthogonal vectors.
    dCl = np.c_[pharynxFunc(xs, *poptX, deriv = 1), pharynxFunc(xs, *poptY, deriv = 1)]#p.diff(cl, axis=0)
    dCl =dCl[:,::-1]
    # normalize northogonal vectors
    dCl[:,0] *=-1
    dClnorm = norm(dCl, axis = 1)
    dCl = dCl/np.repeat(dClnorm[:,np.newaxis], 2, axis =1)
    #dCl = dCl/dClnorm[:,np.newaxis]
    return dCl


def intensityAlongCenterline(im, cl, **kwargs):
    """ Create an intensity kymograph along the centerline.

    Args:
        im (numpy.array): image of a pharynx
        cl (numpy.array or list): (n,2) list of centerline coordinates in image space.
        kwargs: **kwargs are passed skimage.measure.profile_line.

    Returns:
        numpy.array: array of (?,) length. Length is determined by pathlength of centerline.
    """

    if 'width' in kwargs:
        w = kwargs['width']
        kwargs.pop('width', None)
        return np.concatenate([profile_line(im, cl[i], cl[i+1], linewidth = w[i], mode = 'constant', **kwargs) for i in range(len(cl)-1)])
    return np.concatenate([profile_line(im, cl[i], cl[i+1],mode = 'constant', **kwargs) for i in range(len(cl)-1)])


def widthPharynx(cl, contour, dCl):
    """ Use vector interesections to get width of object.
        We are looking for contour points that have the same(or very similar) angle relative to the centerline point as the normal vectors.

    Args:
        cl ([type]): cl (N,2) array describing the centerline
        contour ([type]): (M,2) array describing the contour
        dCl ([type]): (N,2) array describing the normal vectors on the centerline (created by calling .features.normalVecCl(poptX, poptY, xs))

    Returns:
        numpy.array: (N,2) widths of the contour at each centerline point.
    """

    # all possible vectors between contour and centerline
    vCCl = cl[np.newaxis, :] - contour[:,np.newaxis]
    # get normed vectors
    vCClnorm = norm(vCCl, axis = 2)
    vCCl = vCCl/vCClnorm[:,:,np.newaxis]
    # calculate relative angles between centerline and contour-centerline vectors
    angles = np.sum(vCCl*dCl, axis =-1)
    c1 = np.argmin(angles, axis=0)
    c2 = np.argmax(angles, axis=0)
    # new widths
    widths = np.stack([contour[c1], contour[c2]], axis=1)
    return widths


def scalarWidth(widths):
    """calculate the width of the pharynx along the centerline.

    Args:
        widths (numpy.array): (N, 2,2) array of start and end points of lines spanning the pharynx orthogonal to the midline.

    Returns:
        numpy.array: (N,1) array of scalar widtha.
    """
    return np.sqrt(np.sum(np.diff(widths, axis =1)**2, axis =-1))


def straightenPharynx(im, xstart, xend, poptX, poptY, width, nPts = 100):
    """ Based on a centerline, straighten the animal.

    Args:
        im (numpy.array or pims.Frame): image of curved object
        xstart (float): start coordinate to apply to .features._pharynxFunc(x) to create a centerline
        xend (float):  end coordinate to apply to .features._pharynxFunc(x) to create a centerline
        poptX (array): optimal fit parameters describing pharynx centerline.
        poptY (array): optimal fit parameters describing pharynx centerline.
        width (int): how many points to sample orthogonal of the centerline
        nPts (int, optional): how many points to sample along the centerline. Defaults to 100.

    Returns:
        numpy.array: (nPts, width) array of image intensity
    """

    # use linescans to generate straightened animal
    xn = np.linspace(xstart,xend, nPts)
    clF = centerline(poptX, poptY, xn)

    # make vectors orthogonal to the cl
    dCl = normalVecCl(poptX, poptY, xn)
    # create lines intersection the pharynx orthogonal to midline
    widths = np.stack([clF+width*dCl, clF-width*dCl], axis=1)
    # get the intensity profile along these lines
    kymo = [profile_line(im, pts[0], pts[1], linewidth=1, order=3, mode = 'constant') for pts in widths]
    # interpolate to obtain straight image
    tmp = [np.interp(np.arange(-width, width), np.arange(-len(ky)/2, len(ky)/2), ky) for ky in kymo]
    return np.array(tmp)


def gradientPharynx(im):
    """ Apply a local gradient to the image.

    Args:
        im (numpy.array or pims.Frame): image of curved object

    Returns:
        numpy.array: gradient of image
    """

    denoised = rank.median(im, disk(1))
    gradient = rank.gradient(denoised, disk(1))
    return gradient


def extractPump(straightIm):
    """ Use a pumping metric to get measure of pharyngeal contractions.
    It calculates the inverse maximum standard deviation along the Dorsoventral axis.

    Args:
        straightIm (numpy.array): straightened image of pharynx

    Returns:
        float: pharyngeal metric
    """
    return -np.max(np.std(straightIm, axis =1), axis =0)


def headLocationLawn(cl, slices, binLawn):
    """ Use the first coordinate of the centerline to check if the worm touches the lawn.

    Args:
        cl (numpy,array or list): (N,2) centerline spanning the length of the pharynx.
        slice (tuple): (yo, xo) offset between cl and full image
        binLawn ([type]): image of a lawn or other background e.g. created by .features.findLawn

    Returns:
        float: image intensity at first point of cl (should be nose tip)
    """
    y,x = cl[0][0], cl[0][1]
    yo, xo = slices[0], slices[1]
    # make sure that rounding errors don't get you out of bounds
    yn, xn = np.min([binLawn.shape[0]-1, int(y+yo)]), np.min([binLawn.shape[1]-1, int(x+xo)])
    return binLawn[yn, xn]


def inside(x,y,binLawn):
    """Extract intensity of an image at coordinate (x,y).

    Args:
        x (float): x location in px
        y (float): y location in px
        binLawn ([type]): image of a lawn or other background e.g. created by .features.findLawn

    Returns:
        float: image intensity at binLawn(y,x)
    """
    return binLawn[int(y), int(x)]


def calculateImageproperties(df, images):
    """Calculate summary statistics for the padded images.

    Args:
        df (pandas.DataFrame): dataframe with pharaglow results
        images (pims.Stack or numpy.array): stack of N images 

    Returns:
        pandas.DataFrame: dataframe with added columns
    """
    df['Imax'] = np.max(images, axis=(1,2))
    df['Imean'] = np.mean(images, axis=(1,2))
    df['Imedian']= np.median(images, axis=(1,2))
    df['Istd']= np.std(images, axis=(1,2))
    df['skew'] = skew(np.array(images).T.reshape((-1,len(images))))
    return df