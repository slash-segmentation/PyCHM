#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Haar filter written in Cython. This was originally in MEX (MATLAB's Cython-like system) and C++.
Improved speed and added minimal multi-threading.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters.pxi"

from cython.parallel cimport parallel

def cc_cmp_II(ndarray im not None):
    """
    cc_cmp_II
    This function computes the integral image over the input image
    Michael Villamizar
    mvillami@iri.upc.edu
    2009
    Recoded and converted to Cython by Jeffrey Bush 2015-2016.
    Requires im to be a 2-dimensional, behaved, with the last dimension being C-contiguous.
    Returns a newly allocated array.

    input:
        <- Input Image
    output:
        -> Integral Image (II)
    """

    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(double): raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(im, 0), W = PyArray_DIM(im, 1), stride = PyArray_STRIDE(im, 0) // sizeof(double)
    cdef ndarray II_arr = PyArray_EMPTY(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
    cdef DOUBLE_PTR_AR IM = <DOUBLE_PTR_AR>PyArray_DATA(im)
    cdef DOUBLE_PTR_AR II = <DOUBLE_PTR_AR>PyArray_DATA(II_arr)
    cdef DOUBLE_PTR_AR II_last = II

    cdef double last
    cdef intp i, j
    with nogil:
        II[0] = last = IM[0]
        for j in xrange(1, W):
            II[j] = last = IM[j] + last
        for i in xrange(1, H):
            IM += stride
            II += W
            II[0] = last = IM[0] + II_last[0]
            for j in xrange(1, W):
                II[j] = last = IM[j] - II_last[j-1] + last + II_last[j]
            II_last += W
    return II_arr

def cc_Haar_features(ndarray II not None, intp S, ndarray out=None, int nthreads=1, bint compat=True):
    """
    cc_Haar_features
    This function computes the Haar-like features in X and Y
    Michael Villamizar
    mvillami@iri.upc.edu
    2009
    Recoded, multithreaded, and converted to Cython by Jeffrey Bush 2015-2016.
    
    input:
        <- Integral Image (II)
        <- Haar size
    output:
        -> Haar maps : 1) Hx, 2) Hy
    tips:
        * Haar size must be even
    """
    if not PyArray_ISCARRAY_RO(II) or PyArray_TYPE(II) != NPY_DOUBLE or PyArray_NDIM(II) != 2: raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(II,0)-S, II_W = PyArray_DIM(II,1), W = II_W-S
    out = get_out(out, 2, H, W)
    nthreads = get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows
    
    cdef DOUBLE_PTR_AR X = <DOUBLE_PTR_AR>PyArray_DATA(out)             # Haar X output
    cdef DOUBLE_PTR_AR Y = X + PyArray_STRIDE(out, 0) // sizeof(double) # Haar Y output
    cdef DOUBLE_PTR_CAR II_p = <DOUBLE_PTR_CAR>PyArray_DATA(II)
    cdef cc_Haar_features_func cc_Haar_features
    if compat: cc_Haar_features = cc_Haar_features_compat
    else:      cc_Haar_features = cc_Haar_features_core
    
    cdef Range r
    with nogil:
        if nthreads == 1: cc_Haar_features(II_p, H, W, S, X, Y)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(H)
                cc_Haar_features(II_p+r.start*II_W, r.stop-r.start, W, S, X+r.start*W, Y+r.start*W)
    return out

ctypedef void (*cc_Haar_features_func)(DOUBLE_PTR_CAR II, intp H, intp W, intp S, DOUBLE_PTR_AR X, DOUBLE_PTR_AR Y) nogil

cdef void cc_Haar_features_core(DOUBLE_PTR_CAR II, intp H, intp W, intp S, DOUBLE_PTR_AR X, DOUBLE_PTR_AR Y) nogil:
    # Non-compat version doesn't add the arbitary EPS to the area for division (since the area
    # can never be 0 it is pointless). Also the math for X and Y is done in a slightly different
    # order and more efficient (but this causes some minor drift). Overall, this speeds it up by
    # ~25%, the error is ~1e-8 which is less than the EPS value, so overall pretty good.
    cdef intp i, j, C = S//2
    cdef DOUBLE_PTR_CAR T = II         # top
    cdef DOUBLE_PTR_CAR M = II+C*(W+S) # middle
    cdef DOUBLE_PTR_CAR B = II+S*(W+S) # bottom
    cdef double area
    for i in xrange(0, H):
        for j in xrange(0, W):
            area = B[S] + T[0] - T[S] - B[0]
            if area > 0:
                area = 1/area
                X[0] = (B[S] - T[S] - T[0] + B[0] + 2*(T[C] - B[C]))*area
                Y[0] = (B[S] - B[0] - T[0] + T[S] + 2*(M[0] - M[S]))*area
            else:
                X[0] = 0
                Y[0] = 0
            X += 1; Y += 1
            T += 1; M += 1; B += 1
        T += S; M += S; B += S

DEF EPS = 0.00001
cdef void cc_Haar_features_compat(DOUBLE_PTR_CAR II, intp H, intp W, intp S, DOUBLE_PTR_AR X, DOUBLE_PTR_AR Y) nogil:
    cdef intp i, j, C = S//2, WS = W+S
    cdef DOUBLE_PTR_CAR T = II         # top
    cdef DOUBLE_PTR_CAR M = II+C*(W+S) # middle
    cdef DOUBLE_PTR_CAR B = II+S*(W+S) # bottom
    cdef double area
    for i in xrange(0, H):
        for j in xrange(0, W):
            area = B[S] + T[0] - T[S] - B[0]
            if area > 0:
                # NOTE: to be truly compatible, don't pre-compute the division
                # Pre-computing the division doesn't cause any drift (even error ~1e-17) and cuts
                # the time down significantly (~25%) so we keep it
                area = 1/(area+EPS)
                X[0] = ((B[S] + T[C] - T[S] - B[C]) - (B[C] + T[0] - T[C] - B[0]))*area #/(area+EPS)
                Y[0] = ((B[S] + M[0] - M[S] - B[0]) - (M[S] + T[0] - T[S] - M[0]))*area #/(area+EPS)
            else:
                X[0] = 0
                Y[0] = 0
            X += 1; Y += 1
            T += 1; M += 1; B += 1
        T += S; M += S; B += S
