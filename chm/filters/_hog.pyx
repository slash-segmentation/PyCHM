#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
HOG filter written in Cython and C. This was originally in MEX (MATLAB's Cython-like system) and
C++. Improved speed, added multi-threading, and increased accuracy. Additionally, an entirely new
version was created (named hog_new) which runs faster and implements the algorithm much more
accurately than the MATLAB version.

Most of the MATLAB version code is in a separate C++ file. The new Cython version is completely in
this file.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

include "filters.pxi"

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport M_PI, sqrt, atan2
from cython.parallel cimport parallel, prange
from cython.view cimport contiguous

##### MATLAB Compatible Version #####
cdef extern from "HOG.h" nogil:
    cdef intp HOG_init(const intp w, const intp h, const intp *n)
    cdef void HOG_run(const double *pixels, const intp w, const intp h, double *out, double *H, const intp padding)
    cdef intp _HOG "HOG" (const double *pixels, const intp w, const intp h, double *out, const intp n)

def HOG(ndarray[npy_double, ndim=2, mode='c'] pixels not None, ndarray[npy_double, ndim=1, mode='c'] out not None):
    """
    Implements the HOG filter for a single block. Does not support changing the parameters directly
    (they are compiled in). Additionally RGB images are not supported. Instead of allocating the
    memory for the results, you must pass in the output array. The destination array must be a 1-D
    double array and have at least as many elements as are needed for the output. This function
    returns the number of elements stored in the array.
    """
    # NOTE: a much faster method is to bypass this and use "hog_entire(...)"
    if not PyArray_ISBEHAVED_RO(pixels): raise ValueError("Invalid im")
    if not PyArray_ISBEHAVED(out): raise ValueError("Invalid out")
    cdef intp n
    with nogil: n = _HOG(<DOUBLE_PTR_CAR>PyArray_DATA(pixels), PyArray_DIM(pixels, 1), PyArray_DIM(pixels, 0),
                         <DOUBLE_PTR_AR>PyArray_DATA(out), PyArray_DIM(out, 0))
    if n == -1: raise ValueError("Output array too small")
    if n == -2: raise MemoryError()
    return n

##### Highly optimized version for entire image #####
ctypedef void (*filter_func)(DOUBLE_PTR_CAR, const intp, const intp, DOUBLE_PTR_AR, void*, const intp) nogil

cdef bint generic_filter(double[:, :] input,
                         filter_func function, void* data, intp filter_size, intp pad_size,
                         double[:, :, ::contiguous] output) nogil:
    """
    Similar to SciPy's scipy.ndimage.filters.generic_filter with the following new features:
     * The output has one additional dimension over the input and the filter generates an array of
       data for each pixel instead of a scalar
     * The GIL is no longer required, meaning it can be used multi-threaded
     * Instead of extending the edges of the image only the 'valid' pixels are processed
     * Automatic padding added (this could be done by modifying the filter function itself though)

    To simplify the code, the following features were dropped (but could be re-added if necessary):
     * only accepts 2D double matrices as input
     * the last N axes of output must be C-order
     * the footprint is always a square and origin is always zeros
    """
    cdef intp i, j, x
    cdef intp f_rad = filter_size // 2 + pad_size
    cdef intp stride0 = input.strides[0], stride1 = input.strides[1], out_stride = output.strides[0]
    cdef intp full_size = filter_size + pad_size*2
    cdef intp H = input.shape[0] - full_size + 1, W = input.shape[1] - full_size + 1
    cdef intp in_sz = (filter_size + pad_size*2) * (filter_size + pad_size*2), out_sz = output.shape[0]
    
    # Allocate buffers
    cdef INTP_PTR_AR off = <INTP_PTR_AR>malloc(in_sz * sizeof(intp))
    cdef DOUBLE_PTR_AR in_buf = <DOUBLE_PTR_AR>malloc(in_sz * sizeof(double))
    cdef DOUBLE_PTR_AR out_buf = <DOUBLE_PTR_AR>malloc(out_sz * sizeof(double))
    if off is NULL or in_buf is NULL or out_buf is NULL: free(off); free(in_buf); free(out_buf); return False

    # Calculate the offsets
    for i in xrange(full_size):
        for j in xrange(full_size):
            off[i*full_size + j] = (i - f_rad) * stride0 + (j - f_rad) * stride1

    # Get memory pointers
    cdef CHAR_PTR_CA8R pi_row = (<CHAR_PTR_CA8R>&input[0,0]) + f_rad * (stride0 + stride1), pi
    cdef CHAR_PTR_A8R po = <CHAR_PTR_A8R>&output[0,0,0]

    # Process each pixel
    for i in xrange(H):
        pi = pi_row
        for j in xrange(W):
            for x in xrange(in_sz): in_buf[x] = (<DOUBLE_PTR_CAR>(pi + off[x]))[0]
            pi += stride1

            function(in_buf, filter_size, filter_size, out_buf, data, pad_size)

            for x in xrange(out_sz): (<DOUBLE_PTR_AR>(po + x*out_stride))[0] = out_buf[x]
            po += sizeof(double)
        pi_row += stride0

    free(off)
    free(in_buf)
    free(out_buf)

    return True

def hog_entire(ndarray im not None, int filt_width=15, bint compat=True, ndarray out=None, int nthreads=1):
    """
    The entire HOG filter in Cython. Uses a modified scipy.ndimge.filters.generic_filter for
    calling the HOG function. Some other optimizations are using a single memory alocation for
    temporary data storage, giving generic_filter a C function instead of a Python function, and is
    multi-threaded. This also supports running in 'non-compat' which pads the image with image data
    instead of 0s.
    """
    # Check arguments
    if not PyArray_ISBEHAVED_RO(im) or PyArray_TYPE(im) != NPY_DOUBLE or PyArray_NDIM(im) != 2: raise ValueError("Invalid im")
    cdef intp fw_1 = filt_width - 1, pad = 0 if compat else 1
    cdef intp n, H = PyArray_DIM(im, 0) - fw_1 - 2*pad, W = PyArray_DIM(im, 1) - fw_1 - 2*pad
    cdef intp tmp_n = HOG_init(filt_width, filt_width, &n)
    out = get_out(out, n, H, W)
    nthreads = get_nthreads(nthreads, H // 64) # make sure each thread is doing at least 64 rows

    # Setup variables to be given to generic_filter
    cdef bint success = True
    cdef bint* success_p = &success # in OpenMP parallel blocks we cannot directly assign to success, we need a pointer
    cdef double[:,:] im_mv
    if PyArray_ISWRITEABLE(im): im_mv = im
    else:
        # HACK - we never write to im_mv but Cython Memoryviews do not support read-only memory at all
        PyArray_ENABLEFLAGS(im, NPY_ARRAY_WRITEABLE)
        im_mv = im
        PyArray_CLEARFLAGS(im, NPY_ARRAY_WRITEABLE)
    cdef double[:,:,::contiguous] out_mv = out
    
    # Temporary storage (for each thread)
    cdef DOUBLE_PTR_AR tmp = <DOUBLE_PTR_AR>malloc(nthreads * tmp_n * sizeof(double))
    if tmp is NULL: raise MemoryError()
    
    cdef Range r
    with nogil:
        if nthreads == 1:
            success = generic_filter(im_mv, <filter_func>&HOG_run, tmp, filt_width, pad, out_mv)
        else:
            # This uses OpenMP to do the multi-processing
            with parallel(num_threads=nthreads):
                r = get_thread_range(H);
                if not generic_filter(im_mv[r.start:r.stop+fw_1,:], <filter_func>&HOG_run,
                                      tmp+omp_get_thread_num()*tmp_n,
                                      filt_width, pad, out_mv[:,r.start:r.stop,:]):
                    success_p[0] = False

        ## This can be done without OpenMP as well but instead with Python threads, with very little penalty
        #from threading import Thread
        #def thread(intp i):
        #    cdef Range r = get_thread_range(H)
        #    cdef double[:, :] im = im_mv[r.start:r.stop+fw_1,:]
        #    cdef double[:,:,::contiguous] out = out_mv[:,r.start:r.stop,:]
        #    with nogil:
        #        if not generic_filter(im, <filter_func>&HOG_run, tmp+omp_get_thread_num()*tmp_n,
        #                              filt_width, pad, out):
        #            success = False
        #threads = [None] * nthreads
        #for i in xrange(nthreads):
        #    threads[i] = t = Thread(target=thread, name="HOG-"+str(i), args=(i,))
        #    t.start()
        #for t in threads: t.join()
        
    free(tmp)
   
    if not success: raise MemoryError()
    return out

    
##### New Python Version #####
DEF CELL_SIZE=8
DEF BLOCK_SIZE=2
DEF NBINS=9
DEF UNSIGNED_ANGLES=True # both MATLAB and scikits HOG use unsigned angles
DEF NORM='L2-hys' # must be one of 'L2-hys', 'L2-norm', 'L1-norm', or 'L1-sqrt'
# original MATLAB HOG uses L2-hys with a clipping at 0.2 while scikits HOG uses L1-norm
DEF CLIP_VAL=0.2 # only used if NORM is 'L2-hys'
DEF NFEATURES=BLOCK_SIZE*BLOCK_SIZE*NBINS

def hog_new(double[:,::contiguous] im, ndarray out=None, int nthreads=1):
    """
    The HOG filter over the entire image written entirely in Cython. Instead of using a
    generic-filter approach this calculates the histogram for every pixel first then normalizes
    them. This does take more memory as the entire histogram must be stored in memory. This
    temporary memory is approximately im.size*NBINS (with NBINS being 9). The overall speedup is
    around 30-35x faster. This algorithm also generates results closer to the original description
    of the algorithm instead of having several problems that hog_entire has (even when compat is
    False).
    
    Some other changes are that the filter width is fixed at compile-time to 18 which is larger
    than for hog_entire (which is due to the fact that hog_entire isn't actually correctly
    implemented).
    """
    # Get the height and width of the image, the intemediate histogram, and the output
    cdef intp H = im.shape[0], W = im.shape[1]
    cdef intp H_hist = H-CELL_SIZE-1, W_hist = W-CELL_SIZE-1
    cdef intp H_out = H_hist-CELL_SIZE*(BLOCK_SIZE-1), W_out = W_hist-CELL_SIZE*(BLOCK_SIZE-1)
    
    # Make sure each thread is doing at least 64 rows
    nthreads = get_nthreads(nthreads, H_out // 64)
    
    # Allocate the intermediate histogram array and output array
    cdef double[:,:,::1] hist = PyArray_ZEROS(3, [H_hist, W_hist, NBINS], NPY_DOUBLE, False)
    out = get_out(out, NFEATURES, H_out, W_out)
    cdef double[:,:,:] out_mv = out

    # Allocate the temporary blocks used in normalize
    cdef double* blocks = <double*>malloc(NFEATURES*nthreads*sizeof(double))
    if blocks is NULL: raise MemoryError()

    # Allocate the temporary stop positions for dealing with overlaps
    cdef intp* stops = NULL
    if nthreads != 1:
        stops = <intp*>malloc(nthreads*sizeof(intp))
        if stops is NULL: free(blocks); raise MemoryError()

    # Perform the calculations using __gradients and __normalize
    cdef Range r
    cdef intp i
    with nogil:
        if nthreads == 1:
            __gradients(im, 1, H-1, hist)
            __normalize(hist, 0, H_out, blocks, out_mv)
        else:
            # Calculate the gradients except for the CELL_SIZE regions between the chunks
            memset(stops, 0, nthreads*sizeof(intp))
            with parallel(num_threads=nthreads):
                r = get_thread_range(H-2)
                stops[omp_get_thread_num()] = r.stop+1
                __gradients(im, r.start+1, (r.stop if r.stop == (H-2) else r.stop-CELL_SIZE)+1, hist)
            # Calculate the gradiatents in the CELL_SIZE regions between chunks
            for i in prange(nthreads, num_threads=nthreads):
                if stops[i] != 0 and stops[i] != H-1: __gradients(im, stops[i]-CELL_SIZE, stops[i], hist)
            free(stops)
            # Normalize the histogram data
            with parallel(num_threads=nthreads):
                r = get_thread_range(H_out)
                __normalize(hist, r.start, r.stop, blocks + omp_get_thread_num()*NFEATURES, out_mv)

    free(blocks)
    return out
    
cdef void __gradients(double[:,::contiguous] im, intp a, intp b, double[:,:,::1] hist) nogil:
    """
    Calculates all gradients from row a to b in the given image outputing the sum of magnitudes
    for each bin into the histogram.
    
    This function is not thread-safe with respect to the hist argument. It does a += on elements in
    the histogram between a-CELL_SIZE to b. When this is called with several threads with the same
    histogram, the last CELL_SIZE rows of the chunk before must be handled very carefully.
    """
    cdef intp W = im.shape[1], H_hist = hist.shape[0], W_hist = hist.shape[1]
    cdef intp y, x, i, j, i_start, i_end, j_start, j_end, obin
    cdef double mag
    for y in xrange(a, b):
        i_start = max(y-CELL_SIZE, 0)
        i_end   = min(y, H_hist)
        for x in xrange(1, W-1):
            j_start = max(x-CELL_SIZE, 0)
            j_end   = min(x, W_hist)
            obin = __gradient(im, y, x, &mag)
            for i in xrange(i_start, i_end):
                for j in xrange(j_start, j_end):
                    hist[i,j,obin] += mag

cdef inline intp __gradient(double[:,::contiguous] im, intp y, intp x, double* mag) nogil:
    """
    Calculates the gradient for a single pixel in the given image and returns the orientation bin
    along with setting the mag argument.
    """
    cdef double dx = im[y,x+1] - im[y,x-1]
    cdef double dy = im[y+1,x] - im[y-1,x]
    mag[0] = sqrt(dx*dx + dy*dy) # Unnecessary: / (CELL_SIZE * CELL_SIZE)
    cdef double ornt
    IF UNSIGNED_ANGLES:
        ornt = atan2(dy, dx)
        ornt += (ornt<0)*M_PI
        # This is faster but biases taking values from the last bin and places them in the first bin
        #if dx == 0: ornt = 0 if dy == 0 else M_PI/2
        #else: ornt = atan(dy/dx); ornt += (ornt<0)*M_PI
        ornt = ornt*NBINS/M_PI
    ELSE:
        ornt = atan2(dy, dx) + M_PI
        ornt = ornt*NBINS/(2*M_PI)
    cdef intp orientation = <intp>ornt
    if orientation >= NBINS: orientation = NBINS-1
    return orientation

cdef void __normalize(double[:,:,::1] hist, intp a, intp b, double* block, double[:,:,:] out) nogil:
    """
    Normalizes all HOG blocks from rows a to b in the given histogram outputing to out. The
    argument block for temporary memory.
    """
    cdef intp W = out.shape[2], y, x, i, j
    for y in xrange(a, b):
        for x in xrange(W):
            for i in xrange(BLOCK_SIZE):
                for j in xrange(BLOCK_SIZE):
                    memcpy(block+(i*BLOCK_SIZE+j)*NBINS,
                           &hist[y+CELL_SIZE*i, x+CELL_SIZE*j, 0],
                           NBINS*sizeof(double))
            __norm(block)
            for i in xrange(NFEATURES): out[i,y,x] = block[i]

IF NORM == 'L2-hys':
    cdef inline void __norm(double* block) nogil:
        """
        Normalizes a HOG block using the L2-hys norm which normalizes using the L2-norm first,
        clips all values above 0.2 down to 0.2 and then re-normalizes using the L2-norm again.
        """
        cdef intp i
        cdef double norm = 0.0, norm_2 = 0.0
        for i in xrange(NFEATURES): norm += block[i]*block[i]
        if norm != 0.0:
            norm = 1.0 / sqrt(norm)
            for i in xrange(NFEATURES):
                block[i] *= norm
                if block[i] >= CLIP_VAL: block[i] = CLIP_VAL
                norm_2 += block[i]*block[i]
        if norm_2 != 0.0:
            norm_2 = 1.0 / sqrt(norm_2)
            for i in xrange(NFEATURES): block[i] *= norm_2
        else: memset(block, 0, NFEATURES*sizeof(double))
ELIF NORM == 'L2-norm':
    cdef inline void __norm(double* block) nogil:
        """
        Normalizes a HOG block using the L2-norm which divides every value by sqrt(sum(x**2)).
        """
        cdef intp i
        cdef double norm = 0.0
        for i in xrange(NFEATURES): norm += block[i]*block[i]
        if norm != 0.0:
            norm = 1.0 / sqrt(norm)
            for i in xrange(NFEATURES): block[i] *= norm
        else: memset(block, 0, NFEATURES*sizeof(double))
ELSE:
    cdef inline void __norm(double* block) nogil:
        """
        Normalizes a HOG block using either the L1-norm which divides every value by sum(x) or the
        L1-sqrt norm then takes the sqrt of the normalized value.
        """
        cdef intp i
        cdef double norm = 0.0
        for i in xrange(NFEATURES): norm += abs(block[i])
        if norm != 0.0:
            norm = 1.0 / norm
            for i in xrange(NFEATURES):
                IF NORM == 'L1-norm': block[i] *= norm
                ELSE:                 block[i] = sqrt(block[i]*norm) # IF NORM == 'L1-sqrt':
        else: memset(block, 0, NFEATURES*sizeof(double))
