#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: define_macros=NO_IMPORT_ARRAY
"""
Shuffle function written in Cython. Faster than NumPy's shuffle, supports partial shuffling,
and can be used directly from Cython without the GIL.

This does require NumPy since it uses the random-kit RNG bundled with it.

There is a PXD accompanying this file so that the shuffle function can be cimport-ed.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from npy_helper cimport intp

cdef extern from 'randomkit.h' nogil:
    ctypedef struct rk_state:
        pass
    ctypedef enum rk_error:
        pass
    int rk_randomseed(rk_state *state) nogil
    unsigned long rk_interval(unsigned long max, rk_state *state) nogil

# TODO: can these variables be made thread-local?
cdef rk_state rng
cdef bint rng_inited = False

cdef void __init_rng() nogil:
    global rng_inited
    rk_randomseed(&rng)
    rng_inited = True

cpdef void shuffle(intp[::1] arr) nogil:
    """
    Fisher–Yates In-Place Shuffle. It is optimal efficiency and unbiased (assuming the RNG is
    unbiased). This uses the random-kit RNG bundled with Numpy.
    """
    if not rng_inited: __init_rng()
    cdef intp i, j, t, n = arr.shape[0]
    for i in xrange(n-1, 0, -1):
        j = rk_interval(i, &rng)
        t = arr[j]; arr[j] = arr[i]; arr[i] = t

cpdef void shuffle_partial(intp[::1] arr, intp sub) nogil:
    """
    Fisher–Yates In-Place Shuffle. It is optimal efficiency and unbiased (assuming the RNG is
    unbiased). This uses the random-kit RNG bundled with Numpy. Only the LAST `sub` elements
    are shuffled (they are shuffled with the entire array though).
    """
    if not rng_inited: __init_rng()
    cdef intp i, j, t, n = arr.shape[0], end = n-sub-1
    for i in xrange(n-1, end, -1):
        j = rk_interval(i, &rng)
        t = arr[j]; arr[j] = arr[i]; arr[i] = t
    
#from libc.stdlib cimport rand, srand, RAND_MAX
    # The original implementation used rand()/RAND_MAX and was seriously flawed. On some systems
    # any training set with more than 32767 samples (a single 180x180 image would get past that) it
    # would do a divide-by-0 almost every time. Also, the results would be skewed because rand()
    # was not very robust and the method used for scaling the output was also not great.
    #for i in xrange(n-1):
    #   j = i + rand() / (RAND_MAX / (n - i) + 1)
    #   t = arr[j]; arr[j] = arr[i]; arr[i] = t
