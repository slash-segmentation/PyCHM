"""
Haar Filter. Mostly implemented in Cython.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

class Haar(Filter):
    """
    Computes the Haar-like features of the image. This produces Viola and Jones's 2-rectangle
    features (x and y directional features). It uses their method of computing the integral image
    first and then using fast lookups to get the features. The rectangles are 16-pixels on each
    side.

    The compat flag causes slower computations but reduces the drift errors compared to the MATLAB
    output. Additionally, with compat set to False there is no longer an arbitrary value added to
    the area before division (which is unnecessary due to the area never being 0) so it is more
    accurate.

    Uses intermediate memory of O(im.size). While technically it is multi-threaded, it doesn't
    really help much.
    """
    def __init__(self, compat=False):
        super(Haar, self).__init__(8, 2)
        self.__compat = compat

    def __call__(self, im, out=None, region=None, nthreads=1):
        from ._base import get_image_region
        from ._haar import cc_cmp_II, cc_Haar_features #pylint: disable=no-name-in-module
        # OPT: if the following line allocates a new im, the cc_cmp_II could be made to operate in-place
        # Preliminary testing shows ~7.5% speed increase, or ~0.15ms for a 1000x1000 image, so not worth it
        im, region = get_image_region(im, 8, region, nthreads=nthreads)
        ii = cc_cmp_II(im) # INTERMEDIATE: im.shape + (16,16)
        return cc_Haar_features(ii, 16, out, nthreads, self.__compat)
