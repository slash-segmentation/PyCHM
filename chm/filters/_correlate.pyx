#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Optimized correlation filters written in Cython. These were adapated from
scipy.ndimage.correlate1d.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from npy_helper cimport *
import_array()

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport fabs, floor
from libc.float cimport DBL_EPSILON
from cython.parallel cimport parallel
from openmp cimport omp_get_num_threads, omp_get_max_threads, omp_get_thread_num

INPLACE = ndarray((0,0))

def correlate_xy(ndarray im, ndarray weights_x, ndarray weights_y, ndarray out=None, int nthreads=1):
    """
    Run a separable 2D cross correlation, with the filter already separated into weights_x and
    weights_y.
    Equivilent to:
        scipy.ndimage.correlate(im, weights_x[:,None].dot(weights_y[None,:]), output=out)
    With the following differences:
        * edges are dropped so the output is equal to out[r:-r,r:-r]
        * im must be a 2D double, behaved, with a final dimension that is contiguous
        * weights must be 1D double vectors with double-sized final strides
        * this supports multi-threading (although defaults to a single thread)
        * partial in-place operation
    
    The out argument can be one of the following:
        * None, a single array is allocated and used as both the temporary and final output
        * an array that is the width of the input but the height of the output, in which case it is
          used for the temporary and output (no major allocations)
              Note that if the output provided is a view of the input, an additional allocation and
              copy will be made, making it slower than the following two options (which don't have
              the extra copy even when the output is a view of the input)
        * an array that is the output size, in which case an array is allocated for the temporary
        * the special value INPLACE in which case an array is allocated for the temporary but the
          final output is to the input
    """
    cdef ndarray out_x, out_y
    cdef intp[2] dims
    if PyArray_NDIM(im) != 2: raise ValueError("Invalid im")
    if PyArray_NDIM(weights_x) != 1 or PyArray_NDIM(weights_y) != 1: raise ValueError("Invalid weights")
    cdef intp H_in = PyArray_DIM(im, 0), W_in = PyArray_DIM(im, 1)
    cdef intp fs_y = PyArray_DIM(weights_y, 0) - 1, fs_x = PyArray_DIM(weights_x, 0) - 1
    cdef intp H_out = H_in - fs_y, W_out = W_in - fs_x
    if out is None:
        # Temporary and output are allocated as a single array
        dims[0] = H_out; dims[1] = W_in
        out_y = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
        dims[1] = W_out
        out_x = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, PyArray_DATA(out_y))
        Py_INCREF(out_y); PyArray_SetBaseObject(out_x, out_y)
    elif out is INPLACE:
        # Temporary is allocated and final output is to the input
        dims[0] = H_out; dims[1] = W_in
        out_y = PyArray_EMPTY(2, dims, NPY_DOUBLE, False) 
        dims[0] = H_out; dims[1] = W_out
        out_x = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, PyArray_DATA(out))
        Py_INCREF(out); PyArray_SetBaseObject(out_x, out)
    elif PyArray_NDIM(out) != 2: raise ValueError("Invalid output array")
    elif PyArray_DIM(out, 0) == H_out and PyArray_DIM(out, 0) == W_out:
        # Temporary is allocated and final output is to the given output
        dims[0] = H_out; dims[1] = W_in
        out_y = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
        out_x = out
    elif PyArray_DIM(out, 0) == H_out and PyArray_DIM(out, 0) == W_in:
        # The output can be the temporary and is also the final output (no allocation)
        out_y = out
        dims[0] = H_out; dims[1] = W_out
        out_x = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, PyArray_DATA(out))
        Py_INCREF(out); PyArray_SetBaseObject(out_x, out)
    else:
        raise ValueError("Invalid output array")
    correlate_x_or_y(im, weights_y, out_y, nthreads, False)
    return correlate_x_or_y(out_y, weights_x, out_x, nthreads, True)

def correlate_x(ndarray im, ndarray weights, ndarray out=None, int nthreads=1):
    """
    Run cross correlation in 1D along the "x" axis (axis=1) of a 2D image. Equivilent to:
        scipy.ndimage.correlate1d(im, weights, axis=1, output=out)
    With the following differences:
        * edges are dropped so the output is equal to out[:,r:-r]
        * im must be a 2D double, behaved, with a final dimension that is contiguous
        * weights must be a 1D double vector with a double-sized final stride
        * this supports multi-threading (although defaults to a single thread)
        * this supports in-place operation so out can be a view of im (e.g. out=im[:, r:-r] or
          out=im[:, :-2*r] or out=INPLACE)
    """
    return correlate_x_or_y(im, weights, out, nthreads, True)

def correlate_y(ndarray im, ndarray weights, ndarray out=None, int nthreads=1):
    """
    Run cross correlation in 1D along the "y" axis (axis=0) of a 2D image. Equivilent to:
        scipy.ndimage.correlate1d(im, weights, axis=0, output=out)
    With the following differences:
        * edges are dropped so the output is equal to out[r:-r,:]
        * im must be a 2D double, behaved, with a final dimension that is contiguous
        * weights must be a 1D double vector with a double-sized final stride
        * this supports multi-threading (although defaults to a single thread)
    Note: unlike with correlate_x, when operating in-place an internal buffer is allocated for the
    size of the image and the data has to be copied from it to the output, so nothing is saved by
    working in-place.
    """
    return correlate_x_or_y(im, weights, out, nthreads, False)

cdef ndarray correlate_x_or_y(ndarray im, ndarray weights, ndarray out, int nthreads, bint x_dir):
    """
    Runs the correlation in either the X or Y direction. Only some minor differences between the
    two so we share the code.
    """
    from numpy import may_share_memory

    # Check weights
    if not PyArray_ISCARRAY_RO(weights) or PyArray_TYPE(weights) != NPY_DOUBLE or PyArray_NDIM(weights) != 1: raise ValueError("Invalid weights")
    cdef intp filter_size = PyArray_DIM(weights, 0), size1 = filter_size // 2, size2 = filter_size - size1 - 1
    
    # Check im
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2 or PyArray_STRIDE(im, 1) != sizeof(double): raise ValueError("Invalid im")
    cdef intp H = PyArray_DIM(im, 0), W = PyArray_DIM(im, 1)
    if x_dir:
        W -= size1 + size2
        if W < 1: raise ValueError("Filter wider than image")
    else:
        H -= size1 + size2
        if H < 1: raise ValueError("Filter taller than image")
        
    # Limit nthreads
    if nthreads > H // 64: nthreads = H // 64
    if nthreads > omp_get_max_threads(): nthreads = omp_get_max_threads()
    nthreads = 1 if nthreads < 1 else nthreads

    # Check and possibly allocate out
    cdef intp[2] dims
    dims[0] = H; dims[1] = W
    if out is INPLACE:
        out = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, PyArray_DATA(im))
        Py_INCREF(im); PyArray_SetBaseObject(out, im)
    elif out is None:
        out = PyArray_EMPTY(2, dims, NPY_DOUBLE, False)
    elif not PyArray_ISBEHAVED(out) or PyArray_TYPE(out) != NPY_DOUBLE or PyArray_NDIM(out) != 2 or \
         PyArray_DIM(out, 0) != H or PyArray_DIM(out, 1) != W or PyArray_STRIDE(out, 1) != sizeof(double):
        raise ValueError('Invalid output array')
    cdef bint in_place = may_share_memory(im, out)
    cdef ndarray real_out = None
    if in_place:
        if not x_dir: # never possible in-place
            real_out = out
            out = PyArray_EMPTY(2, PyArray_SHAPE(out), NPY_DOUBLE, False)
        elif nthreads > 1:
            # TODO: need some buffer inbetween threads that is done separately
            # in the mean time, do something stupid
            real_out = out
            out = PyArray_EMPTY(2, PyArray_SHAPE(out), NPY_DOUBLE, False)
    
    # Allocate temporary
    cdef double* tmp = <double*>malloc(W * sizeof(double) * nthreads)
    if tmp is NULL: raise MemoryError()

    # Get the pointers
    cdef intp im_stride = PyArray_STRIDE(im, 0) // sizeof(double), out_stride = PyArray_STRIDE(out, 0)
    cdef char* out_p = <char*>PyArray_DATA(out)
    cdef double* in_p = <double*>PyArray_DATA(im) + (size1 if x_dir else (size1 * im_stride))
    cdef double* fw = <double*>PyArray_DATA(weights) + size1

    # Compute the correlation
    cdef correlate_core f
    cdef intp a, b, i
    cdef double inc
    with nogil:
        f = (correlate_x_core if x_dir else correlate_y_core)[__get_sym(fw, size1, size2)]
        if nthreads == 1: f(H, W, out_p, out_stride, in_p, im_stride, tmp, fw, size1, size2)
        else:
            with parallel(num_threads=nthreads):
                nthreads = omp_get_num_threads() # in case there is a difference...
                inc = H / <double>nthreads # a floating point number, use the floor of adding it together
                i = omp_get_thread_num()
                a = <intp>floor(inc*i)
                b = H if i == nthreads-1 else (<intp>floor(inc*(i+1)))
                f(b-a, W, out_p+a*out_stride, out_stride, in_p+a*im_stride, im_stride, tmp+i*W, fw, size1, size2)

    # Free the temporary
    free(tmp)

    # Return the output array
    if real_out is not None:
        PyArray_CopyInto(real_out, out)
        return real_out
    return out

cdef int __get_sym(double* weights, intp size, intp size2) nogil:
    """Checks if a filter is symmetrical (1), anti-symmetrical (2), or other (0)."""
    if size != size2: return 0
    cdef intp i
    for i in xrange(1, size):
        if fabs(weights[i] - weights[-i]) > DBL_EPSILON:
            for i in xrange(1, size+1):
                if fabs(weights[i] + weights[-i]) > DBL_EPSILON:
                    return 0
            return 2
    return 1

    
########## Correlation Core Functions ##########

ctypedef void (*correlate_core)(intp H, intp W,
                                char* out, intp out_stride,
                                const double* in_, intp in_stride,
                                double* tmp, const double* fw, intp size1, intp size2) nogil

cdef correlate_core* correlate_x_core = [correlate_x_any, correlate_x_sym, correlate_x_antisym]
cdef correlate_core* correlate_y_core = [correlate_y_any, correlate_y_sym, correlate_y_antisym]

cdef void correlate_x_sym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double)
    in_stride -= W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k] + in_[-k]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride

cdef void correlate_x_antisym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double)
    in_stride -= W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k] - in_[-k]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride

cdef void correlate_x_any(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double)
    in_stride -= W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[size2] * fw[size2]
            for k in xrange(-size1, size2): tmp[j] += in_[k] * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride

cdef void correlate_y_sym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double), in_stride_rem = in_stride - W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k*in_stride] + in_[-k*in_stride]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride_rem

cdef void correlate_y_antisym(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double), in_stride_rem = in_stride - W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[0] * fw[0]
            for k in xrange(-size1, 0): tmp[j] += (in_[k*in_stride] - in_[-k*in_stride]) * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride_rem

cdef void correlate_y_any(intp H, intp W, char* out, intp out_stride, const double* in_, intp in_stride, double* tmp, const double* fw, intp size1, intp size2) nogil:
    cdef intp i, j, k, nbytes = W * sizeof(double), in_stride_rem = in_stride - W
    for i in xrange(H):
        for j in xrange(W):
            tmp[j] = in_[size2*in_stride] * fw[size2]
            for k in xrange(-size1, size2): tmp[j] += in_[k*in_stride] * fw[k]
            in_ += 1
        memcpy(out, tmp, nbytes)
        out += out_stride
        in_ += in_stride_rem
