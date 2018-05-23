"""
HOG Filter. Mostly implemented in Cython and/or C++.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

class HOG(Filter):
    """
    Computes the HOG (histogram of oriented gradients) features of the image as per Dalal and Bill
    Triggs (2005). This produces 36 features.
    
    This uses cells of 8x8 pixels and blocks of 2x2 cells. Each cell uses 9 orientation bins,
    covering angles from 0 to pi (unsigned). Each block is normalized using the L2-hys norm,
    clipping at 0.2 before renormalizing. The 36 features come from the 9 orientation bins for each
    of the 4 cells per block.

    There are significant differences in compat and non-compat mode. In compat mode this uses a
    15x15 region for each each output which means that the entire 16x16 region is not completely
    covered and an implicit padding of 0s must be added to calculate the final row and column and
    the gradients for the values on the edges.
    
    In non-compat the filter size is instead 18x18 allowing for the entire 16x16 region and the
    gradients along the edge to get all of the data they need. This is much more accurate to the
    original description of the algorithm.
    
    They also take different approaches while caclulating the data. In compat mode each output
    pixel is calculated at once while in non-compat mode the entire histogram is calculated first
    for all pixels then each block is normalized and saved to the output pixel. This has the
    benefit of being much faster (about 30-35x faster) but takes more memory. While the compat mode
    uses minimal intermediate memory the non-compat mode uses O(9*im.size) in temporary memory.
    
    Note that the original MATLAB function used float32 values for many intermediate values so the
    outputs from this function are understandably off by up to 1e-7 even in compat mode.
    """
    def __init__(self, compat=False):
        super(HOG, self).__init__(7 if compat else 9, 36)
        self.__compat = compat

    def __call__(self, im, out=None, region=None, nthreads=1):
        from ._base import get_image_region
        from ._hog import hog_entire, hog_new #pylint: disable=no-name-in-module
        im, region = get_image_region(im, self._padding, region, nthreads=nthreads)
        if self.__compat:
            # TODO: there is a pre-computed division in the C code, should it be kept?
            return hog_entire(im, 15, True, out, nthreads)
        return hog_new(im[1:,1:], out, nthreads) # only needs (8,9) padding
