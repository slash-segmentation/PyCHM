#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Cython code for util. At the moment this includes fast/parallelized copy, hypot, and pooling
functions. Another future function that might go here are a parallelized versions of im2double.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters/filters.pxi"

ctypedef double* dbl_p

from libc.math cimport sqrt, hypot
from cython.view cimport contiguous
from cython.parallel cimport parallel

def par_copy(dst, src, int nthreads):
    """
    Equivilent to numpy.copyto(dst, src, 'unsafe', None) but is parallelized with the given number
    of threads.
    """
    if dst.shape != src.shape: raise ValueError('dst and src must be same shape')
    nthreads = get_nthreads(nthreads, dst.size // 100000000)
    cdef intp N = dst.shape[0]
    cdef Range r
    if nthreads == 1: PyArray_CopyInto(dst, src)
    else:
        with nogil, parallel(num_threads=nthreads):
            r = get_thread_range(N)
            with gil: PyArray_CopyInto(dst[r.start:r.stop], src[r.start:r.stop])

def par_copy_any(dst, src, int nthreads):
    """
    Equivilent to numpy.copyto(dst, src, 'unsafe', None) but is parallelized with the given number
    of threads and works as long as dst and src are the same size (they can be different shapes).
    This is twice as slow as _util.par_copy, so if they are the same shape that should be used.
    """
    if dst.size != src.size: raise ValueError('dst and src must be same size')
    nthreads = get_nthreads(nthreads, dst.size // 50000000)
    cdef intp N1 = dst.shape[0], N2 = src.shape[0]
    cdef Range r1, r2
    if nthreads == 1: PyArray_CopyAnyInto(dst, src)
    else:
        with nogil, parallel(num_threads=nthreads):
            r1 = get_thread_range(N1); r2 = get_thread_range(N2)
            with gil: PyArray_CopyAnyInto(dst[r1.start:r1.stop], src[r2.start:r2.stop])


########## Hypot Function ##########
def par_hypot(ndarray x, ndarray y, ndarray out=None, int nthreads=1, bint precise=False):
    """
    Equivilent to doing sqrt(x*x + y*y) elementwise but slightly faster and parallelized.
    
    If precise is True (default it False), uses the C `hypot` function instead to protect against
    intermediate underflow and overflow situations at the potential cost of time.
    
    This requires the arrays to be 2D doubles with the last axis being contiguous.
    """
    # Check inputs
    cdef intp H = PyArray_DIM(x,0), W = PyArray_DIM(x,1)
    if not PyArray_ISBEHAVED_RO(x) or not PyArray_ISBEHAVED_RO(y) or \
       PyArray_TYPE(x) != NPY_DOUBLE or PyArray_TYPE(y) != NPY_DOUBLE or \
       PyArray_NDIM(x) != 2 or PyArray_NDIM(y) != 2 or H != PyArray_DIM(y, 0) or W != PyArray_DIM(y, 1) or \
       PyArray_STRIDE(x, 1) != sizeof(double) or  PyArray_STRIDE(y, 1) != sizeof(double):
        raise ValueError('Invalid input arrays')
    
    # Check output
    if out is None:
        out = PyArray_EMPTY(2, [H, W], NPY_DOUBLE, False)
    elif not PyArray_ISBEHAVED(out) or PyArray_TYPE(out) != NPY_DOUBLE or PyArray_NDIM(out) != 2 or \
         PyArray_DIM(out, 0) != H or PyArray_DIM(out, 1) != W or PyArray_STRIDE(out, 1) != sizeof(double):
        raise ValueError('Invalid output array')
    
    # Check nthreads
    nthreads = get_nthreads(nthreads, H // 64)
    
    # Get pointers
    cdef intp x_stride = PyArray_STRIDE(x, 0)//sizeof(double), y_stride = PyArray_STRIDE(y, 0)//sizeof(double)
    cdef intp out_stride = PyArray_STRIDE(out, 0)//sizeof(double)
    cdef dbl_p X = <dbl_p>PyArray_DATA(x), Y = <dbl_p>PyArray_DATA(y), OUT = <dbl_p>PyArray_DATA(out)
    
    # Run the hypotenuse calculator
    cdef Range r
    cdef hypot_fp hyp = <hypot_fp>hypot2 if precise else <hypot_fp>hypot1
    with nogil:
        if nthreads == 1: hyp(X, Y, OUT, H, W, x_stride, y_stride, out_stride)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(H)
                hyp(X+r.start*x_stride, Y+r.start*y_stride, OUT+r.start*out_stride, r.stop-r.start,
                    W, x_stride, y_stride, out_stride)
                
    # Done!
    return out

ctypedef void (*hypot_fp)(dbl_p x, dbl_p y, dbl_p out, intp H, intp W, intp x_stride, intp y_stride, intp out_stride) nogil
cdef void hypot1(dbl_p x, dbl_p y, dbl_p out, intp H, intp W, intp x_stride, intp y_stride, intp out_stride) nogil:
    """
    Calculates out = sqrt(x*x + y*y) for each value in x, y, and out. The arrays must all be HxW.
    The strides between rows of each row are given x_stride, y_stride, and out_stride.
    """
    cdef intp i, j
    for i in xrange(H):
        for j in xrange(W): out[j] = sqrt(x[j]*x[j] + y[j]*y[j])
        x += x_stride
        y += y_stride
        out += out_stride
cdef void hypot2(dbl_p x, dbl_p y, dbl_p out, intp H, intp W, intp x_stride, intp y_stride, intp out_stride) nogil:
    """
    Calculates out = hypot(x, y) for each value in x, y, and out (where hypot is the C hypot
    function. The arrays must all be HxW. The strides between rows of each row are given x_stride,
    y_stride, and out_stride.
    """
    cdef intp i, j
    for i in xrange(H):
        for j in xrange(W): out[j] = hypot(x[j], y[j])
        x += x_stride
        y += y_stride
        out += out_stride


########## Max Pooling ##########
# Supported types, can add more as needed
ctypedef fused ptype:
    npy_bool
    npy_double

# Pooling function
ctypedef ptype (*pfunc)(ptype* block, intp stride, intp H, intp W) nogil
# Pooling of 2x2 function
ctypedef ptype (*p2func)(ptype* block, intp stride) nogil

def max_pooling(in_np, int L, ptype[:,::contiguous] out, int nthreads=1):
    """
    Decreases the 2D array size by 2**L. For example if L == 1 the 2D array each dimension is
    halved in size. The downsampling uses max-pooling. The Python function MyMaxPooling should be
    used as they check the inputs, take care of various edge cases (like L==0), allocate the
    output appropiately if necessary, and support the region argument.
    
    Special cases:
        L==1 is made faster (~10%) than without the special code
        booleans is handled specially as well and is faster (~10-25%) than doubles
        
    Currently only supports doubles and booleans but this can be changed at compile-time fairly
    easily. Additionally, a simple change can make it use a different pooling method.
    """
    # Workaround for Cython not supporting readonly buffers
    cdef bint readonly = not PyArray_ISWRITEABLE(in_np)
    if readonly: PyArray_ENABLEFLAGS(in_np, NPY_ARRAY_WRITEABLE)
    cdef ptype[:,::contiguous] in_ = in_np
    if readonly: PyArray_CLEARFLAGS(in_np, NPY_ARRAY_WRITEABLE)
    
    # Get basic values about the blocks and shapes
    cdef intp in_h = in_.shape[0], in_w = in_.shape[1]
    cdef intp H = in_h>>L, W = in_w>>L # size of output not including results from partial blocks
    cdef intp in_stride  = in_.strides[0], out_stride = out.strides[0]
    cdef intp bs = 2<<(L-1), bh = in_h&(bs-1), bw = in_w&(bs-1) # size of standard or partial blocks
    nthreads = get_nthreads(nthreads, H // 128)
    cdef pfunc f = pool_max[ptype] # the pooling function
    cdef p2func f2 = pool2_max[ptype] # the pooling function for 2x2 blocks
    
    # Pool the data
    cdef Range r
    cdef intp y, Y, h
    with nogil:
        if nthreads == 1:
            # Core of the data
            if bs == 2: pool2[ptype](f2, &in_[0,0], &out[0,0], in_stride, out_stride, H, W)
            else: pool[ptype](f, &in_[0,0], &out[0,0], in_stride, out_stride, H, W, bs, bs)
            # Partial column
            if bw > 0: pool[ptype](f, &in_[0,W*bs], &out[0,W], in_stride, out_stride, H, 1, bs, bw)
        else:
            with parallel(num_threads=nthreads):
                r = get_thread_range(H)
                y = r.start; Y = y*bs; h = r.stop-y
                # Core of the data
                if bs == 2: pool2[ptype](f2, &in_[Y,0], &out[y,0], in_stride, out_stride, h, W)
                else: pool[ptype](f, &in_[Y,0], &out[y,0], in_stride, out_stride, h, W, bs, bs)
                # Partial column
                if bw > 0: pool[ptype](f, &in_[Y,W*bs], &out[y,W], in_stride, out_stride, h, 1, bs, bw)
        if bh > 0:
            # Partial row
            pool[ptype](f, &in_[H*bs,0], &out[H,0], in_stride, out_stride, 1, W, bh, bs)
            # Partial column and row
            if bw > 0: out[H,W] = f(&in_[H*bs, W*bs], in_stride, bh, bw)
    return out

cdef inline void pool(pfunc f, ptype *in_, ptype *out, intp in_stride, intp out_stride, intp H, intp W, intp bh, intp bw) nogil:
    """
    Pool the data from in_ to out using the function f. The size of each block is given by (bh,bw).
    The size of out is given by (H,W). The size of in_ is not explicity given.
    """
    cdef intp x, y
    for y in xrange(H):
        for x in xrange(W):
            out[x] = f(in_+x*bh, in_stride, bh, bw)
        in_ = <ptype*>((<char*>in_)+in_stride*bh)
        out = <ptype*>((<char*>out)+out_stride)

cdef inline ptype pool_max(ptype* block, intp stride, intp H, intp W) nogil:
    """Calculates the max of a HxW block."""
    cdef intp j
    cdef ptype mx

    if ptype is npy_bool:
        # Special case for booleans that does logical-or with short-circuiting
        for _ in xrange(H):
            for j in xrange(W):
                if block[j]: return True
            block = <ptype*>((<char*>block)+stride)
        return False
    
    else:
        mx = block[0]
        for j in xrange(1, W):
            if block[j] > mx: mx = block[j]
        block = <ptype*>((<char*>block)+stride)
        for _ in xrange(1, H):
            for j in xrange(W):
                if block[j] > mx: mx = block[j]
            block = <ptype*>((<char*>block)+stride)
        return mx

cdef inline void pool2(p2func f, ptype *in_, ptype *out, intp in_stride, intp out_stride, intp H, intp W) nogil:
    """
    Pool the data from in_ to out using the function f with the size of each block is (2,2). The
    size of out is given by (H,W). The size of in_ is not explicity given.
    """
    cdef intp x, y
    for y in xrange(H):
        for x in xrange(W):
            out[x] = f(in_+x*2, in_stride)
        in_ = <ptype*>((<char*>in_)+in_stride*2)
        out = <ptype*>((<char*>out)+out_stride)

cdef inline ptype pool2_max(ptype* block, intp stride) nogil:
    """Calculates the max of a 2x2 block."""
    cdef intp j
    cdef ptype mx

    if ptype is npy_bool:
        # Special case for booleans that does logical-or with short-circuiting
        return (block[0] or block[1] or (<ptype*>((<char*>block)+stride))[0] or
                (<ptype*>((<char*>block)+stride))[1])
    
    else:
        mx = block[1] if (block[1] > block[0]) else block[0]
        block = <ptype*>((<char*>block)+stride)
        if block[0] > mx: mx = block[0]
        if block[1] > mx: mx = block[1]
        return mx
