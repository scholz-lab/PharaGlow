#!/usr/bin/env python

"""features.py: image analysis of pharynx. Uses skimage to provide image functionality."""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skimage.filters import threshold_li
from skimage.morphology import medial_axis, skeletonize
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese, inverse_gaussian_gradient,checkerboard_level_set
from skimage.transform import rescale
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import curve_fit

from skimage.measure import find_contours, profile_line

def thresholdPharynx(im):
	"""use li threshold to obtain mask of pharynx.
		input: image of shape (N,M)
		output: binary image (N,M)."""
	return mask = im>threshold_li(data[k])


def skeletonPharynx(mask):
	"""use skeletonization to obatain midline of pharynx.
		input: binary mask (N, M) 
		output: skeleton (N,M)"""
	return skeletonize(mask)


def sortSkeleton(skeleton):
	"""Use hierarchical clustering with optimal ordering to get \
		the best path through the skeleton points.
		input is a skeleton"""
	# coordinates of skeleton
	ptsX, ptsY = np.where(skeleton)
	# cluster
	Z = linkage(np.c_[ptsX, ptsY], method='average', metric='cityblock', optimal_ordering=True)
	return leaves_list(Z)


def _pharynxFunc(x, *p, deriv = 0):
	"""cubic polynomial helper function"""
    if deriv==1:
        return p[1] + 2*p[2]*x
    return p[0] + p[1]*x + p[2]*x**2


def fitSkeleton(ptsX, ptsY)
	"""Fit a (cubic) polynomial spline to the centerline. The input should be sorted skeleton coordinates.
	"""
	nP = len(ptsX)
	x = np.arange(nP)
	# fit each axis separately
	poptX, pcov = curve_fit(_pharynxFunc, x, ptsX, p0=(1,1,1))
	poptY, pcov = curve_fit(_pharynxFunc, x, ptsY, p0 = (1,1,1))
	
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
	contour = np.array(contour/scale).T
	return contour


def cropcenterline(poptX, poptY, contour, nP):
	"""Define start and end point of centerline by crossing of contour. 
	Inputs: poptX, poptY optimal fit parameters describing pharynx shape/centerline.
			contour: (N,2) array of points describing the pharynx outline.
	output: start and end coordinate to apply to _pharynxFunc(x) to create a centerline 
	spanning the length of the pharynx.."""
	xs = np.linspace(-0.25*nP,1.25*nP, 100)
	tmpcl = np.c_[func(xs, *poptX), func(xs, *poptY)]
	# update centerline based on crossing the contour
	# we are looking for two crossing points
	distClC = np.sum((tmpcl-contour[:,np.newaxis])**2, axis =-1)
	start, end = np.argsort(np.min(distClC, axis = 0))[:2]

	# update centerline length
	xstart, xend = xs[start],xs[end]
	xs = np.linspace(xs[start],xs[end], nP)
	cl = np.c_[func(xs, *poptX), func(xs, *poptY)]
	return xs[start],xs[end]


def centerline(poptX, poptY, xs):
	"""create a centerline from fitted function.
		Inputs: poptX, poptY optimal fit parameters describing pharynx shape/centerline.
		xs: array of coordinates to create centerline from _pharynxFunc(x, *p, deriv = 0).
		output: (N,2) acenterline spanning the length of the pharynx.. Same length as xs.
		"""
	return np.c_[func(xs, *poptX), func(xs, *poptY)]



def normalVecCl(poptX, poptY, xs):
	"""create vectors normal to the centerline by using the derivative of the function describing the midline.
	inputs: poptX, poptY optimal fit parameters describing pharynx shape/centerline.
			xs: array of coordinates to create centerline from _pharynxFunc(x, *p, deriv = 0).
	output: (N,2) array of unit vectors orthogonal to centerline. Same length as xs.
	"""

	# make an orthogonal vector to the cl by calculating derivative (dx, dy) and using (-dy, dx) as orthogonal vectors.
	dCl = np.c_[func(xs, *poptY, deriv = 1), func(xs, *poptX, deriv = 1)]#p.diff(cl, axis=0)
	dCl =dCl[:,::-1]
	# normalize northogonal vectors
	dCl[:,0] *=-1
	dClnorm = norm(dCl, axis = 1)
	dCl = dCl/dClnorm[:,np.newaxis]
	return dCl


def intensityAlongCenterline(im, cl, **kwargs):
	"""create a kymograph along the centerline. 
		inputs: im: grayscale image
				cl (n,2) list of centerline coordinates in image space.
		kwargs: **kwargs are passed skimage.measure.profile_line.

		output: array of (?,) length. Length is determined by pathlength of centerline.
		"""
	if 'width' in kwargs:
		return np.concatenate([profile_line(im, cl[i], cl[i+1], lw = kwargs['width'][i],**kwargs) for i in range(len(cl)-1)])
	return np.concatenate([profile_line(im, cl[i], cl[i+1], **kwargs) for i in range(len(cl)-1)])


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
	print(vCCl.shape, dCl.shape)
	# calculate relative angles between centerline and contour-centerline vectors
	angles = np.sum(vCCl*dCl, axis =-1)
	print(angles.shape)
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


def straightenPharynx(im, xstart, xend, poptX, poptY, width):
	"""Based on centerline, straighten the animal."""
	# use linescans to generate straightened animal
	xn = np.linspace(xstart,xend, 50)
	clF = centerline(poptX, poptY, xn)
	
	# make vectors orthogonal to the cl
	dCl = normalVecCl(poptX, poptY, xn)
	# create lines intersection the pharynx orthogonal to midline
	widths = np.stack([clF+width*dCl, clF-width*dCl], axis=1)
	# get the intensity profile along these lines
	kymo = [profile_line(im, pts[0], pts[1], linewidth=1, order=3) for pts in widths]
	# interpolate to obtain straight image
	tmp = [np.interp(np.arange(-width, width), np.arange(-len(ky)/2, len(ky)/2), ky) for ky in kymo]
	returm tmp