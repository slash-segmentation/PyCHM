#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: language=c++

"""
The core imresize functions in Cython. Just 2 functions are exposed: imresize and imresize_fast.
Each function is specialized for C or Fortran contiguous with a fallback to any strided array. They
support all basic numpy numerical types.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from npy_helper cimport *
import_array()
include "fused.pxi"

from libc.math cimport floor
from cython.view cimport contiguous
from cython.parallel cimport prange, parallel
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

def imresize(ndarray im not None, ndarray out not None, ndarray weights not None, ndarray indices not None, int nthreads=1):
    """
    Internal function for imresize written in Cython. The image must be aligned, RxC, and an
    integral or floating-point type (but not float16). The output array must be ((R+1)//2)xC and
    the same data type as the image.  The weights and indices must be RxN and with the final
    dimension contiguous. They must be of type double and intp respectively. Not all of the above
    requirements are actually checked, but bad things will happen if they are not followed. Also,
    there will be speed-ups if some additional requirements are met, like the image and output both
    being C or Fortran contiguous.
    """
    cdef bint fortran = PyArray_ISFARRAY_RO(im) and PyArray_ISFARRAY(out)
    cdef bint c_cont  = PyArray_ISCARRAY_RO(im) and PyArray_ISCARRAY(out)
    nthreads = max(min(nthreads, PyArray_SIZE(im) / 262144, omp_get_max_threads()), 1)
    if fortran:  __imresize_f[PyArray_TYPE(im)](im, out, weights, indices, nthreads)
    elif c_cont: __imresize_c[PyArray_TYPE(im)](im, out, weights, indices, nthreads)
    else:        __imresize_s[PyArray_TYPE(im)](im, out, weights, indices, nthreads)

@fused
def __imresize_s(npy_number_basic[:,:] im, npy_number_basic[:,:] out, double[:,::contiguous] weights, Py_ssize_t[:,::contiguous] indices, int nthreads):
    """Strided specialization wrapper, needed to work well with fused types"""
    cdef intp a, b, i, Rout = weights.shape[0]
    cdef double inc
    with nogil:
        if nthreads == 1: __imresize_s_core(im, out, weights, indices, 0, Rout)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = Rout / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = Rout if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                __imresize_s_core(im, out, weights, indices, a, b)

@fused
def __imresize_f(npy_number_basic[::1,:] im, npy_number_basic[::1,:] out, double[:,::contiguous] weights, Py_ssize_t[:,::contiguous] indices, int nthreads):
    """Fortran specialization wrapper, needed to work well with fused types"""
    cdef intp a, b, i, C = im.shape[1]
    cdef double inc
    with nogil:
        if nthreads == 1: __imresize_f_core(im, out, weights, indices, 0, C)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = C / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = C if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                __imresize_f_core(im, out, weights, indices, a, b)

@fused
def __imresize_c(npy_number_basic[:,::1] im, npy_number_basic[:,::1] out, double[:,::contiguous] weights, Py_ssize_t[:,::contiguous] indices, int nthreads):
    """C specialization wrapper, needed to work well with fused types"""
    cdef intp a, b, i, Rout = weights.shape[0]
    cdef double inc
    with nogil:
        if nthreads == 1: __imresize_c_core(im, out, weights, indices, 0, Rout)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = Rout / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = Rout if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                __imresize_c_core(im, out, weights, indices, a, b)

cdef void __imresize_s_core(npy_number_basic[:,:] im, npy_number_basic[:,:] out,
                            double[:,::contiguous] weights, Py_ssize_t[:,::contiguous] indices,
                            intp r_start, intp r_end) nogil:
    """imresize core for strided memory layout"""
    cdef intp C = im.shape[1], N = weights.shape[1], r, c, n
    cdef double* weights_p
    cdef intp*   indices_p
    cdef double value
    for r in xrange(r_start, r_end):
        weights_p = &weights[r,0]
        indices_p = <intp*>&indices[r,0]
        for c in xrange(C):
            value = 0.0
            for n in xrange(N): value += weights_p[n] * im[indices_p[n],c]
            out[r,c] = cast_with_clip(value, <npy_number_basic>0)

cdef void __imresize_f_core(npy_number_basic[::1,:] im, npy_number_basic[::1,:] out,
                            double[:,::contiguous] weights, Py_ssize_t[:,::contiguous] indices,
                            intp c_start, intp c_end) nogil:
    """imresize core for Fortran memory layout"""
    cdef intp Rout = weights.shape[0], N = weights.shape[1], r, c, n
    cdef npy_number_basic* im_p
    cdef npy_number_basic* out_p
    cdef double* weights_p
    cdef intp*   indices_p
    cdef double value
    for c in xrange(c_start, c_end):
        im_p  = &im[0,c]
        out_p = &out[0,c]
        for r in xrange(Rout):
            weights_p = &weights[r,0]
            indices_p = <intp*>&indices[r,0]
            value = 0.0
            for n in xrange(N): value += weights_p[n] * im_p[indices_p[n]]
            out_p[r] = cast_with_clip(value, <npy_number_basic>0)

cdef void __imresize_c_core(npy_number_basic[:,::1] im, npy_number_basic[:,::1] out,
                            double[:,::contiguous] weights, Py_ssize_t[:,::contiguous] indices,
                            intp r_start, intp r_end) nogil:
    """imresize core for C contiguous memory layout"""
    cdef intp C = im.shape[1], N = weights.shape[1], r, c, n
    cdef npy_number_basic* out_p
    cdef double* weights_p
    cdef intp*   indices_p
    cdef double value
    for r in xrange(r_start, r_end):
        out_p = &out[r,0]
        weights_p = &weights[r,0]
        indices_p = <intp*>&indices[r,0]
        for c in xrange(C):
            value = 0.0
            for n in xrange(N): value += weights_p[n] * im[indices_p[n],c]
            out_p[c] = cast_with_clip(value, <npy_number_basic>0)


########## Fast imresize ##########
# These are optimized in several way over the general methods
# First, they use a staticly allocated set of weights which shaves off a lot of time and memory
# Second, the indices can be calculated as needed which means lots of memory is saved along with some time
# Third, it can be known that exactly two rows on the top and bottom need clipping of the indices, which saves even more time

def imresize_fast(ndarray im not None, ndarray out not None, int nthreads=1):
    """
    Internal function for imresize_fast written in Cython. The image and output have the same
    requirements as per imresize.
    """
    cdef bint fortran = PyArray_ISFARRAY_RO(im) and PyArray_ISFARRAY(out)
    cdef bint c_cont  = PyArray_ISCARRAY_RO(im) and PyArray_ISCARRAY(out)
    nthreads = max(min(nthreads, PyArray_SIZE(im) / 262144, omp_get_max_threads()), 1)
    if fortran:  __imresize_fast_f[PyArray_TYPE(im)](im, out, nthreads)
    elif c_cont: __imresize_fast_c[PyArray_TYPE(im)](im, out, nthreads)
    else:        __imresize_fast_s[PyArray_TYPE(im)](im, out, nthreads)

@fused
def __imresize_fast_s(npy_number_basic[:,:] im, npy_number_basic[:,:] out, int nthreads):
    """Fast-strided specialization wrapper, needed to work well with fused types"""
    cdef intp a, b, i, Rout = (im.shape[0] + 1) // 2
    cdef double inc
    with nogil:
        # Do the top and bottom 2 rows first
        __imresize_fast_TB(im, out)
        
        # Now the bulk of the data
        if nthreads == 1: __imresize_fast_s_core(im, out, 2, Rout-2)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = (Rout-4) / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = (Rout-4) if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                __imresize_fast_s_core(im, out, a+2, b+2)

@fused
def __imresize_fast_f(npy_number_basic[::1,:] im, npy_number_basic[::1,:] out, int nthreads):
    """Fast-Fortran specialization wrapper, needed to work well with fused types"""
    cdef npy_number_basic[:,:] im_s = im, out_s = out
    cdef intp a, b, i, C = im.shape[1]
    cdef double inc
    with nogil:
        # Do the top and bottom 2 rows first
        __imresize_fast_TB(im_s, out_s)
        
        # Now the bulk of the data
        if nthreads == 1: __imresize_fast_f_core(im, out, 0, C)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = C / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = C if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                __imresize_fast_f_core(im, out, a, b)

@fused
def __imresize_fast_c(npy_number_basic[:,::1] im, npy_number_basic[:,::1] out, int nthreads):
    """Fast-C specialization wrapper, needed to work well with fused types"""
    cdef npy_number_basic[:,:] im_s = im, out_s = out
    cdef intp a, b, i, Rout = (im.shape[0] + 1) // 2
    cdef double inc
    with nogil:
        # Do the top and bottom 2 rows first
        __imresize_fast_TB(im_s, out_s)
        
        # Now the bulk of the data
        if nthreads == 1: __imresize_fast_c_core(im, out, 2, Rout-2)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = (Rout-4) / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = (Rout-4) if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                __imresize_fast_c_core(im, out, a+2, b+2)

cdef double* bicubic_weights = [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875]

cdef void __imresize_fast_TB(npy_number_basic[:,:] im, npy_number_basic[:,:] out) nogil:
    """imresize_fast top and bottom 2 rows for any array, no need to optimize for just 4 rows..."""
    cdef intp Rin = im.shape[0], C = im.shape[1], Rout = (Rin + 1) // 2, r, c, n, i_start
    cdef double value
    
    # The first two rows
    for r in xrange(min(2, Rout)):
        i_start = -3+2*r
        for c in xrange(C):
            value = 0.0
            for n in xrange(8): value += bicubic_weights[n] * im[min(max(i_start+n, 0), Rin-1),c]
            out[r,c] = cast_with_clip(value, <npy_number_basic>0)

    # The last two rows
    for r in xrange(max(Rout-2, 2), Rout):
        i_start = -3+2*r
        for c in xrange(C):
            value = 0.0
            for n in xrange(8): value += bicubic_weights[n] * im[min(i_start+n, Rin-1),c]
            out[r,c] = cast_with_clip(value, <npy_number_basic>0)

cdef void __imresize_fast_s_core(npy_number_basic[:,:] im, npy_number_basic[:,:] out, intp r_start, intp r_end) nogil:
    """imresize_fast core for strided memory layout"""
    Rout = (im.shape[0] + 1) // 2
    cdef intp C = im.shape[1], r, c, n, i_start
    cdef double value
    for r in xrange(r_start, r_end):
        i_start = -3+2*r
        for c in xrange(C):
            value = 0.0
            for n in xrange(8): value += bicubic_weights[n] * im[i_start+n,c]
            out[r,c] = cast_with_clip(value, <npy_number_basic>0)

cdef void __imresize_fast_f_core(npy_number_basic[::1,:] im, npy_number_basic[::1,:] out, intp c_start, intp c_end) nogil:
    """imresize_fast core for Fortran memory layout"""
    cdef intp Rout = (im.shape[0] + 1) // 2, r, c, n, i_start
    cdef npy_number_basic* im_c
    cdef npy_number_basic* out_c
    cdef double value
    for c in xrange(c_start, c_end):
        im_c  = &im[0,c]
        out_c = &out[0,c]
        for r in xrange(2, Rout-2):
            value = 0.0
            i_start = -3+2*r
            for n in xrange(8): value += bicubic_weights[n] * im_c[i_start + n]
            out_c[r] = cast_with_clip(value, <npy_number_basic>0)

cdef void __imresize_fast_c_core(npy_number_basic[:,::1] im, npy_number_basic[:,::1] out, intp r_start, intp r_end) nogil:
    """imresize_fast core for C contiguous memory layout"""
    cdef intp C = im.shape[1], r, c, n, i_start
    cdef npy_number_basic* out_r
    cdef npy_number_basic* im_rc
    cdef double value
    for r in xrange(r_start, r_end):
        out_r = &out[r,0]
        i_start = -3+2*r
        for c in xrange(C):
            value = 0.0
            im_rc = &im[i_start,c]
            for n in xrange(8): value += bicubic_weights[n] * im_rc[n*C]
            out_r[c] = cast_with_clip(value, <npy_number_basic>0)
