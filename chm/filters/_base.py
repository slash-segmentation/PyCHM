"""
Base for filters. Includes the base class, the aggregate filter class, along with utility functions.
Also intalls the Cython handler.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod

# Make a bunch of utils function available from here
from ..utils import get_image_region, replace_sym_padding, hypot, copy, copy_flat, next_regular #pylint: disable=unused-import

__all__ = ['Filter', 'FilterBank']

########## Filter Base Class ##########
class Filter(object):
    """
    Base filter class. A filter is an operation on an image that generates 1 or more image-sized
    features that represent the original image in some way. The filter may require some extra
    padding to be valid.

    Besides the properties "padding", "features", and "should_normalize", every filter has a single
    operation which is invoked simply by calling the filter, giving it the image along with
    optional output, region, number of threads.
    
    Some filters take an optional parameter when being constructing to operate in "compatibility
    mode" with the original MATLAB mode.
    """
    __metaclass__ = ABCMeta

    def __init__(self, padding, features):
        self._padding = padding
        self._features = features

    @abstractmethod
    def __call__(self, im, out=None, region=None, nthreads=1):
        """
        Runs the filter on the image. The image should be a float64 2D array. Some filters may have
        more stringent requirements like C-ordered.

        Optional arguments:
            out      array to store the results in instead of allocating it, if None it is allocated;
                     the output is always returned even if it isn't allocated; the output must be 3D
                     with the first dimension equal to self.features and the last two dimensions equal
                     to the shape of the image
            region   describes the region of the image to process (as a sequence of top, left, bottom,
                     right) that allows a filter to possibly use the neighboring pixel data instead of
                     padding the image, if None then padding is added if necessary
            nthreads the number of threads to use while computing the filter, defaulting to a single
                     thread; not all filters support multi-threading so won't benefit from it
        """
        pass

    @property
    def padding(self):
        """The padding needed by this filter"""
        return self._padding

    @property
    def features(self):
        """The number of features created by this filter"""
        return self._features
    
    @property
    def should_normalize(self):
        """
        A sequence of True/False values for each features if that features should be normalized when
        requested (default implementation returns True for each feature).
        """
        return (True,)*self.features


########## Cumulative Filter ##########
class FilterBank(Filter):
    """
    An aggregate filter. It takes a sequence of many filters and runs all of them in-order. It has
    some optimizations in that it pre-computes/allocates the regioned-image and output. The nthreads
    argument is simply passed along to the other filters.
    """
    def __init__(self, filters):
        super(FilterBank, self).__init__(max(f.padding for f in filters), sum(f.features for f in filters))
        self.__filters = tuple(filters)
    @property
    def filters(self): return self.__filters
    @property
    def should_normalize(self): return sum((f.should_normalize for f in self.__filters), ())
    def __call__(self, im, out=None, region=None, nthreads=1):
        from ..utils import im2double, set_lib_threads
        set_lib_threads(nthreads)
        
        # Pad/region the image
        P = self._padding
        im = im2double(get_image_region(im, P, region, nthreads=nthreads)[0])
        region = (P, P, im.shape[0]-P, im.shape[1]-P)

        # Get the output for the filters
        if out is None:
            from numpy import empty
            out = empty((self._features, im.shape[0] - 2*P, im.shape[1] - 2*P))

        # Run the filters
        nf_start = 0
        for f in self.__filters:
            nf = f.features
            f(im, out=out[nf_start:nf_start+nf], region=region, nthreads=nthreads)
            nf_start += nf
        
        return out


########## Utilities ##########
def separate_filter(f):
    """
    This takes a 2D convolution filter and separates it into 2 1D filters (vertical and horizontal).
    If it can't be separated then the original filter will be returned.
    """
    from numpy import finfo, sqrt
    from numpy.linalg import svd
    u, s, v = svd(f)
    tol = s[0] * max(f.shape) * finfo(s.dtype).eps
    if (s[1:] > tol).any(): return f
    scale = sqrt(s[0])
    return u[:,0]*scale, v[0,:]*scale

def round_u8_steps(arr):
    """
    Clips the data between 0.0 to 1.0 then rounds the data to steps of 1/255. This is to match the
    behavior of MATLAB's imfilter being run on uint8 data. All of this is done in-place on the data
    in the array.
    """
    from numpy import clip
    clip(arr, 0.0, 1.0, out=arr)
    arr *= 255.0
    arr.round(out=arr)
    arr /= 255.0

def run_threads(func, total, min_size=1, nthreads=1):
    """
    Runs a bunch of threads, over "total" items, chunking the items. The function is given a
    "threadid" (a value 0 to nthreads-1, inclusive) along with a start and stop index to run over
    (where stop is not included). This mirrors the similar Cython function.
    """
    from multiprocessing import cpu_count
    from threading import Thread
    from math import floor
    nthreads = min(nthreads, (total + (min_size // 2)) // min_size, cpu_count())
    if nthreads <= 1:
        func(0, 0, total)
    else:
        inc = total / nthreads
        inds = [0] + [int(floor(inc*i)) for i in xrange(1, nthreads)] + [total]
        threads = [Thread(target=func, args=(i, inds[i], inds[i+1])) for i in xrange(nthreads)]
        for t in threads: t.start()
        for t in threads: t.join()
