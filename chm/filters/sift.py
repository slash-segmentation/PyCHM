"""
SIFT Filter. Partially implemented in Cython for speed and multi-threading.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

class SIFT(Filter):
    def __init__(self, compat=False):
        super(SIFT, self).__init__(SIFT.__filter_padding, SIFT.__filter_features)
        self.__compat = compat
    
    def __call__(self, im, out=None, region=None, nthreads=1):
        from scipy.ndimage.filters import correlate1d
        from ._base import get_image_region, replace_sym_padding, round_u8_steps
        if self.__compat:
            # In compatibility mode, instead of symmetric reflections we pad with 0s
            im, region = replace_sym_padding(im, 7, region, 15, nthreads)
        else:
            im, region = get_image_region(im, 15, region, nthreads=nthreads)
        gauss_filter = SIFT.__gauss_filter
        im = correlate1d(im, gauss_filter, 0, mode='nearest') # INTERMEDIATE: im.shape
        im = correlate1d(im, gauss_filter, 1, mode='nearest') # TODO: TEMPORARY: im.shape
        if self.__compat: round_u8_steps(im)
        im = SIFT.__dense_sift(im, None, nthreads)
        region = (region[0]-7, region[1]-7, region[2]-7, region[3]-7)
        im = im[:, region[0]:region[2], region[1]:region[3]]
        if out is not None:
            from numpy import copyto
            copyto(out, im)
        return im

    @staticmethod
    def __dense_sift(im, out, nthreads):
        # CHANGED: forced grid_spacing = 1 and patch_size is now a constant defined globally
        # CHANGED: the input im can be modified slightly in place (scaling)
        # CHANGED: no longer returns optional grid
        # CHANGED: a few pieces have been removed and made into Cython functions for speed and multi-threading
        from numpy import empty, empty_like, sqrt, arctan2, cos, sin
        from scipy.ndimage.filters import correlate1d
        from ._base import run_threads
        from ._sift import orientations, neighborhoods, normalize #pylint: disable=no-name-in-module

        # TODO: don't precompute division?
        # TODO: this sometimes causes a divide-by-zero
        im *= 1/im.max() # can modify im here since it is always the padded image

        H, W = im.shape

        dgauss_filter = SIFT.__dgauss_filter
        tmp = empty_like(im) # INTERMEDIATES: im.shape

        # vertical edges
        correlate1d(im, dgauss_filter[1], 0, mode='constant', output=tmp)
        imx = correlate1d(tmp, dgauss_filter[0], 1, mode='constant') # INTERMEDIATE: im.shape
        
        # horizontal edges
        correlate1d(im, dgauss_filter[0], 0, mode='constant', output=tmp)
        imy = correlate1d(tmp, dgauss_filter[1], 1, mode='constant') # INTERMEDIATE: im.shape

        im_theta = arctan2(imy, imx, out=tmp)
        im_cos, im_sin = cos(im_theta), sin(im_theta, out=tmp) # INTERMEDIATE: im.shape (cos)
        del im_theta, tmp
        
        imx *= imx; imy *= imy; imx += imy
        im_mag = sqrt(imx, out=imx) # gradient magnitude
        del imx, imy # cleanup 1 intermediate (imy)
        
        num_angles, num_bins, patch_sz = SIFT.__num_angles, SIFT.__num_bins, SIFT.__patch_sz
        im_orientation = orientations(im_mag, im_cos, im_sin, num_angles, nthreads=nthreads)
        del im_mag, im_cos, im_sin # cleanup 3 intermediates

        def thread(_, start, stop):
            weight_x, weight_x_origin = SIFT.__weight_x, SIFT.__weight_x_origin
            tmp = empty_like(im) # INTERMEDIATE: im.shape * nthreads
            for a in xrange(start, stop):
                #pylint: disable=unsubscriptable-object
                correlate1d(im_orientation[a,:,:], weight_x, 0, mode='constant', origin=weight_x_origin, output=tmp)
                correlate1d(tmp, weight_x, 1, mode='constant', origin=weight_x_origin, output=im_orientation[a,:,:])
        # OPT: these take about 12% of the time of SIFT (when single-threaded)
        run_threads(thread, num_angles)
        
        H, W = H-patch_sz+2, W-patch_sz+2
        if out is None:
            out = empty((num_bins*num_bins*num_angles, H, W), dtype=im.dtype)

        # OPT: takes about 19% of time in SIFT (when single-threaded)
        sift_arr = out.reshape((num_bins*num_bins, num_angles, H, W))
        neighborhoods(sift_arr, im_orientation, num_bins, patch_sz//num_bins, nthreads=nthreads)
        #del im_orientation # cleanup large intermediate
        im_orientation = None # cannot delete since it is used in a nested scope, this should work though
        
        # normalize SIFT descriptors
        # OPT: takes about 55% of time in SIFT (when single-threaded)
        normalize(out.reshape((num_bins*num_bins*num_angles, H*W)), nthreads=nthreads)
        return out


    ##### Static Fields #####
    __gauss_filter = None
    __dgauss_filter = None
    __patch_sz = 16 # must be even (actually should be a multiple of num_bins I think)
    __padding_sz = (__patch_sz // 2) - 1 + 3 + 2
    __num_angles = 8 # if num_angles or num_bins is changed, it is likely other things have to change as well
    __num_bins = 4
    #__alpha = 9 # parameter for attenuation of angles (must be odd) [hard coded]
    __weight_x = None
    __weight_x_origin = None
    __filter_padding = 15 #(__patch_sz // 2) - 1 TODO: dynamically calculate this again
    __filter_features = __num_bins*__num_bins*__num_angles

    @staticmethod
    def __gen_gauss(sigma):
        # CHANGED: no longer deals with anisotropic Gaussians or 2 sigmas
        from numpy import ceil
        f_wid = 4 * ceil(sigma) + 1
        return SIFT.__fspecial_gauss(f_wid, sigma)

    @staticmethod
    def __gen_dgauss(sigma):
        # CHANGED: same changes as to __gen_gauss
        # CHANGED: only returns one direction of gradient, not both
        # laplacian of size sigma
        from numpy import gradient
        G, _ = gradient(SIFT.__gen_gauss(sigma))
        G *= 2
        G /= abs(G).sum()
        return G

    @staticmethod
    def __fspecial_gauss(shape, sigma):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',shape,sigma)
        """
        # CHANGED: unlike MATLAB's fspecial('gaussian', ...) this does not accept a tuple for shape/sigma
        from numpy import ogrid, exp, finfo
        n = (shape-1)/2
        y,x = ogrid[-n:n+1,-n:n+1]
        h = exp(-(x*x+y*y)/(2*sigma*sigma))
        h[h < finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0: h /= sumh
        return h

    @staticmethod
    def __static_init__():
        from numpy import arange
        from ._base import separate_filter
        
        SIFT.__gauss_filter = separate_filter(SIFT.__fspecial_gauss(7, 1.0))[0]
        SIFT.__gauss_filter.flags.writeable = False
        
        SIFT.__dgauss_filter = separate_filter(SIFT.__gen_dgauss(1.0))
        SIFT.__dgauss_filter[0].flags.writeable = False
        SIFT.__dgauss_filter[1].flags.writeable = False

        # Convolution formulation
        # Note: the original has an apparent bug where the weight_x vector is not centered (has 2 extra zeros on right side)
        nb = SIFT.__num_bins
        bins_patch = nb/SIFT.__patch_sz
        ## Original definition (lots of leading and trailing zeros, 2 extra trailing 0s)
        #weight_x = 1 - abs(arange(1, patch_sz+1) - (half_patch_sz//2) + 0.5)*bins_patch
        #weight_x *= (weight_x > 0)
        ## No leading zeros, 2 trailing zeros, compatible with original definition and slighlty faster and needing less padding
        #weight_x = 1 - abs(arange(1, nb-1, bins_patch) - 0.5*(nb-bins_patch))
        #weight_x = concatenate((weight_x, [0,0]))
        ## No trailing/leading zeros (if this is used origin=weight_x_origin is needed to make it compatible)
        SIFT.__weight_x = 1 - abs(arange(1, nb-1, bins_patch) - 0.5*(nb-bins_patch))
        SIFT.__weight_x.flags.writeable = False
        SIFT.__weight_x_origin = -2 # correlate origin, for convolution it needs to be 1

SIFT.__static_init__()
