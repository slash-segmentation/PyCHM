#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
SIFT filter parts implemented in Cython. SIFT was originally in MATLAB but various steps were
converted to Cython because it was slow and can benefit from multi-threading.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters.pxi"

from libc.math cimport sqrt, cos, sin, M_PI
from cython.parallel cimport prange
from cython cimport view

def orientations(double[:,::1] im_mag not None,
                 double[:,::1] im_cos not None,
                 double[:,::1] im_sin not None, int num_angles, int nthreads=1):
    # CHANGED: this was extracted from dense_sift for speed increase and multi-threading
    cdef intp a, i, j, H = im_cos.shape[0], W = im_cos.shape[1]
    cdef double x, angle_cos, angle_sin, angle_step = 2*M_PI/num_angles
    
    # make orientation images
    # for each histogram angle
    cdef ndarray out = PyArray_EMPTY(3, [num_angles, H, W], NPY_DOUBLE, False) # INTERMEDIATE: 8 * im.shape
    cdef double[:,:,::1] im_orientation = out
    for a in prange(num_angles, nogil=True, schedule='static', num_threads=nthreads):
        # compute each orientation channel
        angle_cos = cos(a*angle_step)
        angle_sin = sin(a*angle_step)
        for i in xrange(H):
            for j in xrange(W):
                x = im_cos[i,j] * angle_cos + im_sin[i,j] * angle_sin
                x = x * (x > 0)
                #x **= __sift.alpha # yes, power is REALLY slow, better to do it by hand...
                x = x * x * x # x^3
                x = x * x * x # x^9
                # weight by magnitude
                im_orientation[a,i,j] = x * im_mag[i,j]
    return out

def neighborhoods(double[:,:,:,::view.contiguous] out not None,
                  double[:,:,::1] im_orientation not None, int num_bins, int step, int nthreads=1):
    # CHANGED: this was extracted from dense_sift for speed increase and multi-threading
    cdef intp i, j, x, y, H = out.shape[2], W = out.shape[3]
    for i in prange(num_bins, nogil=True, schedule='static', num_threads=nthreads):
        x = i*step
        for j in xrange(num_bins):
            y = j*step
            out[i*num_bins+j,:,:,:] = im_orientation[:,y:y+H,x:x+W]

def normalize(double[:,::view.contiguous] sift_arr not None, int nthreads=1):
    """normalize SIFT descriptors (after Lowe)"""
    # CHANGED: this operates on sift_arr in place and operates multi-threaded
    # OPT: this would work better with the C-order transposed array... but that would likely need to copy the data
    # However, that may not help, I did try copying columns of sift_arr to a temporary as necessary
    # and moving back at the end and it hurt the time.

    cdef intp i, j, nfeat = sift_arr.shape[0], npix = sift_arr.shape[1]
    cdef double norm

    # find indices of descriptors to be normalized (those whose norm is larger than 1)
    for j in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
    # or if we plan on having a lot of things hit that 'continue'
    #for j in prange(npix, nogil=True, schedule='guided', num_threads=nthreads):
        norm = 0
        for i in xrange(nfeat): norm += sift_arr[i,j] * sift_arr[i,j]
        if norm <= 1: continue
        # TODO: don't precompute division? (here and below)
        norm = 1/sqrt(norm)
        for i in xrange(nfeat): 
            sift_arr[i,j] *= norm
            if sift_arr[i,j] > 0.2: sift_arr[i,j] = 0.2 # suppress large gradients
        # finally, renormalize to unit length
        norm = 0
        for i in xrange(nfeat): norm += sift_arr[i,j] * sift_arr[i,j]
        norm = 1/sqrt(norm)
        for i in xrange(nfeat): sift_arr[i,j] *= norm
