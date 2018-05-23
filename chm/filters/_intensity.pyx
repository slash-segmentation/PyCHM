#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Intensity / neighborhood filter written in Cython. This was converted because it was so heavily
used and benefited from the multi-threading.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters.pxi"

from libc.string cimport memcpy
from cython.parallel cimport prange

def intensity(ndarray im, ndarray offsets, ndarray out=None, tuple region=None, int nthreads=1):
    """
    Computes the intensity/neighborhood features of the image. See the Python wrapper for more
    information. Here the offsets must be of intp type.
    """
    # TODO: all DOUBLE_PTR_R and DOUBLE_PTR_CR should be aligned (A) but doing so causes illegal argument core dumps...

    # Check image and offsets
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(double): raise ValueError("Invalid im")
    if not PyArray_ISBEHAVED_RO(offsets) or PyArray_TYPE(offsets) != NPY_INTP or PyArray_NDIM(offsets) != 2 or PyArray_DIM(offsets, 0) != 2 or PyArray_STRIDE(offsets, 1) != sizeof(intp): raise ValueError("Invalid offsets")
    cdef intp i, n_offs = PyArray_DIM(offsets, 1), im_stride = PyArray_STRIDE(im, 0)
    
    # Get the image region and offset radii
    cdef intp H, W, T, L, B, R, T_rad, L_rad, B_rad, R_rad
    if region is None:
        T, L, B, R = 0, 0, 0, 0 # everything outside needs reflection
        H, W = PyArray_DIM(im, 0), PyArray_DIM(im, 1)
    else:
        T, L, B, R = region
        H, W = B - T, R - L
        B, R = PyArray_DIM(im, 0) - B, PyArray_DIM(im, 1) - R
    L_rad, T_rad = -PyArray_Min(offsets, 1, NULL)
    R_rad, B_rad =  PyArray_Max(offsets, 1, NULL)
    if L_rad < 0 or R_rad < 0 or T_rad < 0 or B_rad < 0: raise ValueError("Offsets cannot be purely positive or negative")

    # Make sure we have somewhere to save to
    out = get_out(out, n_offs, H, W)
    cdef intp out_stride = PyArray_STRIDE(out, 0)

    # Get the image rows, including the reflected ones
    cdef ndarray im_rows = PyArray_EMPTY(1, [H+T_rad+B_rad], NPY_INTP, False)
    cdef const char** irp = <const char**>PyArray_DATA(im_rows)
    cdef const char* im_p = <const char*>PyArray_DATA(im)
    cdef pre_y  = clip_neg( T_rad-T)
    cdef off_y  = clip_neg(-T_rad+T)
    cdef post_y = clip_neg( B_rad-B)
    cdef mid_y  = H - pre_y - post_y + T_rad + B_rad + off_y
    for i in xrange(pre_y-1, -1, -1):             irp[0] = im_p+i*im_stride; irp += 1 # reflected start
    for i in xrange(off_y, mid_y):                irp[0] = im_p+i*im_stride; irp += 1
    for i in xrange(mid_y-1, mid_y-post_y-1, -1): irp[0] = im_p+i*im_stride; irp += 1 # reflected end

    # Get pointers
    cdef const DOUBLE_PTR_CR* im_rows_p = (<const DOUBLE_PTR_CR*>PyArray_DATA(im_rows)) + T_rad
    cdef const intp* xs = <const intp*>PyArray_DATA(offsets)
    cdef const intp* ys = xs + PyArray_STRIDE(offsets, 0) // sizeof(intp)
    cdef char* out_p = <char*>PyArray_DATA(out)
    R = L_rad-R_rad-R # TODO: test this when L_rad != R_rad to make sure this is right...

    # Copy every offset
    nthreads = get_nthreads(nthreads, H // 64)
    for i in prange(n_offs, nogil=True, schedule='static', num_threads=nthreads):
        cpy_off(<DOUBLE_PTR_R>(out_p + i*out_stride), im_rows_p + ys[i], W, H, xs[i], L, R)
    return out

cdef inline void cpy_off(DOUBLE_PTR_R out, const DOUBLE_PTR_CR* im, const intp W, const intp H, const intp X, const intp L, const intp R) nogil:
    cdef intp i, j
    cdef intp pre_x  = clip_neg(-X-L)
    cdef intp off_x  = clip_neg( X+L)
    cdef intp post_x = clip_neg( X+R)
    cdef intp mid_x  = W - pre_x - post_x
    cdef intp end_x  = mid_x-post_x-1
    cdef DOUBLE_PTR_CR row
    for i in xrange(H):
        row = im[0]; im = im + 1
        for j in xrange(pre_x-1, -1, -1): out[0] = row[j]; out += 1 # first few reflected
        row += off_x; memcpy(out, row, mid_x*sizeof(double)); out += mid_x # middle straight copied
        for j in xrange(mid_x-1, end_x, -1): out[0] = row[j]; out += 1 # last few reflected

cdef inline intp clip_neg(intp x) nogil: return 0 if x < 0 else x
