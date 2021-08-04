#!/usr/bin/env python

"""features.py: image analysis of pharynx. Uses skimage to provide image functionality."""
import pims
import numpy as np
from numpy.linalg import norm
from skimage import util
from skimage.filters import threshold_li, threshold_yen, gaussian
from skimage.morphology import skeletonize, watershed, disk, remove_small_holes, remove_small_objects
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import morphological_chan_vese, inverse_gaussian_gradient,checkerboard_level_set
from skimage.transform import rescale
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import curve_fit
from skimage.filters import rank, threshold_li, gaussian, threshold_otsu
from skimage.measure import find_contours, profile_line, regionprops, label



def findLawn(image, smooth = 1, areaHoles = 15000, areaSpots = 50000):
    """binarize the image of the bacterial lawn."""
    image = gaussian(image, smooth, preserve_range = True)
    thresh = threshold_otsu(image)
    binary = image > thresh
    binary = remove_small_holes(binary, area_threshold=areaHoles, connectivity=1, in_place=False)
    binary = remove_small_objects(binary, min_size=areaSpots, connectivity=8, in_place=False)
    return binary

@pims.pipeline
def thresholdPharynx(im):
    """use li threshold to obtain mask of pharynx.
        input: image of shape (N,M) 
        output: binary image (N,M).
    """
    mask = im>threshold_yen(im)
    labeled = label(mask)
    # keep only the largest item
    area = 0
    for region in regionprops(labeled):
        if area <= region.area:
            mask = labeled==region.label
            area = region.area
    return mask


def skeletonPharynx(mask):
    """use skeletonization to obatain midline of pharynx.
        input: binary mask (N, M) 
        output: skeleton (N,M)"""
    return skeletonize(mask)

@pims.pipeline
def sortSkeleton(skeleton):
    """Use hierarchical clustering with optimal ordering to get \
        the best path through the skeleton points.
        input: skeletonized image
        output: sorted points on skeleton.
        """
    # coordinates of skeleton
    ptsX, ptsY = np.where(skeleton)
    # cluster
    Z = linkage(np.c_[ptsX, ptsY], method='average', metric='cityblock', optimal_ordering=True)
    return leaves_list(Z)


def pharynxFunc(x, *p, deriv = 0):
    """defines a cubic polynomial helper function"""
    if deriv==1:
        return p[1] + 2*p[2]*x
    return p[0] + p[1]*x + p[2]*x**2


def fitSkeleton(ptsX, ptsY, func = pharynxFunc):
    """Fit a (cubic) polynomial spline to the centerline. The input should be sorted skeleton coordinates.
        ptsX: sorted x coordinate
        ptsY: sorted y coordinate
        func: Fit function, by default a cubic polynomial.
        output: optimal fit parameters of pharynxFunc
    """
    nP = len(ptsX)
    x = np.arange(nP)
    # fit each axis separately
    poptX, pcov = curve_fit(func, x, ptsX, p0=(np.mean(ptsX),1,1))
    poptY, pcov = curve_fit(func, x, ptsY, p0 = (np.mean(ptsY),1,1))
    
    return poptX, poptY


def morphologicalPharynxContour(mask, scale = 4, **kwargs):
    """use morphological contour finding on the mask image to get a nice outline.
        We will upsample the image to get more exact outlines.
        **kwargs are handed to morphological_chan_vese.
        input: binary mask of pharynx.
        output: coordinates of the contour as array of (N,2) coordinates."""
    
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


def cropcenterline(poptX, poptY, contour, nP):
    """Define start and end point of centerline by crossing of contour. 
    Inputs: poptX, poptY optimal fit parameters describing pharynx shape/centerline.
            contour: (N,2) array of points describing the pharynx outline.
            nP: number of points in original skeleton.
    output: start and end coordinate to apply to _pharynxFunc(x) to create a centerline 
    spanning the length of the pharynx.."""
    xs = np.linspace(-0.25*nP,1.25*nP, 100)
    tmpcl = np.c_[pharynxFunc(xs, *poptX), pharynxFunc(xs, *poptY)]
    # update centerline based on crossing the contour
    # we are looking for two crossing points
    distClC = np.sum((tmpcl-contour[:,np.newaxis])**2, axis =-1)
    start, end = np.argsort(np.min(distClC, axis = 0))[:2]

    # update centerline length
    xstart, xend = xs[start],xs[end]
    return xs[start],xs[end]


def centerline(poptX, poptY, xs):
    """create a centerline from fitted function.
        Inputs: poptX, poptY optimal fit parameters describing pharynx shape/centerline.
        xs: array of coordinates to create centerline from _pharynxFunc(x, *p, deriv = 0).
        output: (N,2) acenterline spanning the length of the pharynx. Same length as xs.
        """
    return np.c_[pharynxFunc(xs, *poptX), pharynxFunc(xs, *poptY)]



def normalVecCl(poptX, poptY, xs):
    """create vectors normal to the centerline by using the derivative of the function describing the midline.
    inputs: poptX, poptY optimal fit parameters describing pharynx shape/centerline.
            xs: array of coordinates to create centerline from _pharynxFunc(x, *p, deriv = 0).
    output: (N,2) array of unit vectors orthogonal to centerline. Same length as xs.
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
    """create a kymograph along the centerline. 
        inputs: im: grayscale image
                cl (n,2) list of centerline coordinates in image space.
        kwargs: **kwargs are passed skimage.measure.profile_line.

        output: array of (?,) length. Length is determined by pathlength of centerline.
        """
    if 'width' in kwargs:
        w = kwargs['width']
        kwargs.pop('width', None)
        return np.concatenate([profile_line(im, cl[i], cl[i+1], linewidth = w[i], mode = 'constant', **kwargs) for i in range(len(cl)-1)])
    return np.concatenate([profile_line(im, cl[i], cl[i+1],mode = 'constant', **kwargs) for i in range(len(cl)-1)])


def widthPharynx(cl, contour, dCl):
    """Use vector interesections to get width of object. 
        We are looking for contour points that have the same(or very similar) angle relative to the centerline point as the normal vectors".
        inputs: cl (N,2) array
                contour (M,2) array
                dCl (N,2) array (can be created by calling normalVecCl(poptX, poptY, xs))
        outputs: (N,2) widths of the contour at each centerline point.
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
        input: (N, 2,2) array of start and end points of lines 
        spanning the pharynx orthogonal to the midline.
        output: (N,1) array of scalar width."""
    return np.sqrt(np.sum(np.diff(widths, axis =1)**2, axis =-1))


def straightenPharynx(im, xstart, xend, poptX, poptY, width, nPts = 100):
    """Based on centerline, straighten the animal.
    input: 
    im: an image
    xstart, xend, poptX, poptY are the parameters of a curve/centerline describing the shape of the pharynx
    width: how far to sample left and right of the centerline
    nPts: how any points to sample along the centerline
    output: (nPts, width) array of image intensity
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
    """apply a local gradient to the image.
        input:
            im: image (M,N)
        output: gradient of image (M,N)
    """
    #im = util.img_as_ubyte(im)
    denoised = rank.median(im, disk(1))
    gradient = rank.gradient(denoised, disk(1))
    return gradient#util.img_as_ubyte(gradient)


def extractPump(straightIm):
    """use pumping metric to get measure of bulb contraction. It calculates the inverse maximum standard deviation along the anteriorposterior axis.
        input: straightened images of a pharynx (M,N,T)
        output: pharyngeal metric (T,)
    """
    return -np.max(np.std(straightIm, axis =1), axis =0)


def headLocationLawn(cl, slice, binLawn):
    """use the first coordinate of the centerline to check if the worm touches the lawn.
        Inputs:
            cl: (N,2) centerline spanning the length of the pharynx.
            slice: (yo, xo) offset between cl and full image
            binLawn: image of a lawn or other background
        Outputs:
            Intensity at first point of cl (should be nose tip)
    """
    y,x = cl[0][0], cl[0][1]
    yo, xo = slice[0], slice[1]
    # make sure that rounding errors don't get you out of bounds
    yn, xn = np.min([binLawn.shape[0]-1, int(y+yo)]), np.min([binLawn.shape[1]-1, int(x+xo)])
    return binLawn[yn, xn]


def inside(x,y,binLawn):
    """Extract intensity of an image at coordinate (x,y).
        Inputs:
            x,y: location in px
            binLawn: image of a lawn or other background
        Outputs:
            Intensity at coordinate
    """
    return binLawn[int(y), int(x)]

