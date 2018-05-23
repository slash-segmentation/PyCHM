#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""
CHM LDNN functions written in Cython. These includes parts of kmeansML, distSqr,
UpdateDiscriminants, and UpdateDiscriminants_SB (last two are renamed though).

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from npy_helper cimport *
import_array()

from cython.parallel cimport prange

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp, sqrt
from libc.float cimport DBL_MAX
ctypedef double* dbl_p

from chm.__shuffle cimport shuffle, shuffle_partial

#################### General ####################

def downsample(intp min, double[:,:] X, char[::1] Y, intp downsample=10):
    """
    The full data set is in X and Y specifies which rows in X to use. If the number of Trues in Y is
    small, all of those rows in X are used. If it is large then most samples are skipped according
    to `downsample`.
    
    This code is equivilent to:
        from numpy import flatnonzero, empty
        n_trues = Y.sum()
        if n_trues < min:
            raise ValueError('Not enough data - either increase the data size or lower the number of levels (%d < %d)' % (n_trues,k))
        if n_trues <= min*downsample or downsample == 1:
            out = empty((n_trues, X.shape[0]))
            if X.flags.c_contiguous: X.compress(Y, 1, out.T)
            else:                    X.T.compress(Y, 0, out)
        else:
            out = empty((((n_trues+downsample-1)//downsample), X.shape[0]))
            idx = flatnonzero(Y)[::downsample]
            if X.flags.c_contiguous: X.take(idx, 1, out.T)
            else:                    X.T.take(idx, 0, out)
        return out

    However this is ~1.25x (for C ordered X) to ~2.25x (for F ordered X) faster and uses no temporary
    memory allocations. This is ~6x faster when X is Fortran-ordered but even on C-ordered data this
    is significantly faster than copying the X data to a new array with Fortran ordering.

    Inputs:
        min         minimum number of samples to keep
        X           m-by-n double matrix
        Y           n length boolean vector (given as chars since Cython does not understand boolean type)
        downsample  the amount of downsampling to do, defaults to 10

    Returns:
        X'          n*-by-m C-ordered matrix (where n* is the selected/downsampled n)
    """
    cdef intp m = X.shape[0], n = X.shape[1], n_trues = 0, skip = downsample, i, j = 0
    
    assert(Y.shape[0] == n)
    assert(downsample >= 1)

    # Count the number of trues: n_trues = Y.sum()
    for i in xrange(n): n_trues += <bint>Y[i] # get the number of True values
    if n_trues < min:
        raise ValueError('Not enough data - either increase the data size or lower the number of levels (%d < %d)' % (n_trues,min))
    
    # X.take(flatnonzero(Y), 1)  (possibly downsampled)
    cdef bint ds = downsample > 1 and n_trues > min*downsample
    cdef intp n_out = (n_trues+downsample-1)//downsample if ds else n_trues
    cdef double[:,::1] data = PyArray_EMPTY(2, [n_out, m], NPY_DOUBLE, False)
    if ds:
        for i in xrange(n):
            if Y[i]:
                if skip == downsample: data[j,:] = X[:,i]; j += 1; skip = 0
                skip += 1
    else:
        for i in xrange(n):
            if Y[i]: data[j,:] = X[:,i]; j += 1
    return data.base

def stddev(double[:,::1] X):
    """
    Calculates the standard deviation of an n-by-m 2D C-array across axis 0. Equivalent to:
        np.std(X, 0)
    except it is significantly faster (~4x in testing) and uses O(m) extra memory unlike
    the Numpy function which uses O(n*m) memory.
    """
    # I also tested a parallel version of this and with 2 threads it was ~15% faster and
    # barely benefited from a third or more thread. It made the code significantly more
    # complex and even for very large datasets that is ~10ms right before a process that
    # is going to take minutes so it isn't worth it.
    cdef intp N = X.shape[0], M = X.shape[1], i, j
    cdef double[::1] stds = PyArray_ZEROS(1, &M, NPY_DOUBLE, False)
    cdef double[::1] means = PyArray_ZEROS(1, &M, NPY_DOUBLE, False)
    with nogil:
        for i in xrange(N):
            for j in xrange(M): means[j] += X[i,j]
        for j in xrange(M): means[j] = means[j] / N
        for i in xrange(N):
            for j in xrange(M): stds[j] += (X[i,j]-means[j])*(X[i,j]-means[j])
        for j in xrange(M): stds[j] = sqrt(stds[j]/N)
    return stds.base

cdef inline double clip(double x, double mn, double mx) nogil:
    """Clip/clamp a number x inbetween the values mn and mx."""
    # Note: on GCC and Clang this hopefully procudes code using SSE maxsd and minsd assembly which
    # is really fast. See https://stackoverflow.com/a/16659263/582298
    return mn if x < mn else (mx if x > mx else x)

def run_kmeans(intp k, X, Y, intp downsmpl=10, intp repeats=5, bint whiten=False, int nthreads=1):
    """
    Downsample, possibly 'whiten', and run k-means on the data.

    Inputs:
        X        m-by-n matrix where m is the number of features and n is the number of samples
        Y        n-length array of bool labels
        
    Parameters:
        k        number of clusters to make
        downsmpl amount of downsampling to perform on the data before clustering
        repeats  times to repeat running k-means looking for a better solution
        whiten   scale each feature vector in X to have a variance of 1 before running k-means

    Returns:
        means   k-by-m matrix, cluster means/centroids
    """
    # NOTE: This could be easily outside of Cython but it calls 3 Cython functions in a row and that is about it...
    from numpy import int8
    
    assert(X.ndim == 2 and Y.ndim == 1 and X.shape[1] == Y.shape[0] and Y.dtype == bool and X.dtype == float)
    assert(downsmpl >= 1 and repeats >= 1)
    
    # Downsample the data (always creates a transposed copy that is C-ordered)
    data = downsample(k, X, Y.view(int8), downsmpl)
    
    # 'Whiten' the data (make variance of each feature equal to 1)
    if whiten:
        sd = stddev(data)
        sd[sd < 1e-10] = 1 # avoid divide-by-0 and blowing-up near-no-variance data by not scaling them
        # TODO: the check above could be relative to the mean for that feature, which is already calculated when calculating the stddev
        data *= 1/sd
    
    # Calculate clusters using kmeans
    # OPT: parallelize this
    clusters,rms2 = kmeansML(k, data)
    for _ in xrange(1, repeats):
        new_clusters,new_rms2 = kmeansML(k, data)
        if new_rms2 < rms2:
            clusters = new_clusters
            rms2 = new_rms2
    
    # Un-whiten the clusters
    if whiten: clusters *= sd

    return clusters


#################### K-Means ML ####################

# This requires SciPy 0.16.0 released in Aug 2015
from scipy.linalg.cython_blas cimport dgemm

DEF KMEANS_ML_MAX_ITER      = 100
DEF KMEANS_ML_ETOL          = 0.0
DEF KMEANS_ML_DTOL          = 0.0

DEF KMEANS_ML_RATE          = 3
DEF KMEANS_ML_MIN_N         = 50

cdef bint kmeans_high_acc = False # makes k-means default to the slower but more accurate distSqr_acc function

class ClusteringWarning(UserWarning): pass

cpdef kmeansML(intp k, double[:,::1] data):
    """
    Mulit-level K-Means. Tries very hard to always return k clusters. The parameters are now
    compile-time constants and no longer returns membership or RMS error, however these could be
    added back fairly easily (they are available in this function, just not returned).

    Parameter values:
        maxiter 100
        dtol    0.0
        etol    0.0

    Inputs:
        k       number of clusters
        data    n-by-d matrix where n is the number of samples and d is the features per sample

    Returns:
        means   k-by-d matrix, cluster means/centroids
        rms2    RMS^2 error of the clusters

    Originally by David R. Martin <dmartin@eecs.berkeley.edu> - October 2002

    Jeffrey Bush, 2015-2017, NCMIR, UCSD
    Converted into Python/Cython and optimized greatly
    """
    # Allocate memory
    cdef intp n = data.shape[0], d = data.shape[1]
    cdef ndarray means      = PyArray_EMPTY(2, [k, d], NPY_DOUBLE, False)
    cdef ndarray means_next = PyArray_EMPTY(2, [k, d], NPY_DOUBLE, False)
    cdef ndarray membership = PyArray_EMPTY(1, &n, NPY_INTP, False)
    cdef ndarray counts     = PyArray_EMPTY(1, &k, NPY_INTP, False)
    cdef ndarray temp       = PyArray_EMPTY(2, [k, n], NPY_DOUBLE, False)
    
    # Run k-means
    global kmeans_high_acc
    cdef bint orig_high_acc = kmeans_high_acc
    cdef double rms2 = __kmeansML(k, data, means, means_next, membership, counts, temp, orig_high_acc)
    if not orig_high_acc and rms2 == -1.0: # When ran into an accuracy issue, need to increase accuracy
        kmeans_high_acc = True
        rms2 = __kmeansML(k, data, means, means_next, membership, counts, temp, True)
    if rms2 == -1.0: raise RuntimeError('K-Means error: RMS^2 should always decrease and not be negative')
    
    # All done
    return means, rms2

cdef double __kmeansML(intp k, double[:,::1] data, double[:,::1] means, double[:,::1] means_next,
                       intp[::1] membership, intp[::1] counts, double[:,::1] temp, bint high_acc=False) except -2.0:
    """
    Mulit-level K-Means core. Originally the private function kmeansInternal.

    Inputs:
        k           number of clusters
        data        n-by-d matrix where n is the number of samples and d is the features per sample
        high_acc    if True then use distSqr_acc which is slower but higher accuracy (default False)
        
    Outputs:
        means       k-by-d matrix, cluster means/centroids
        means_next  k-by-d matrix used as a temporary
        membership  n length vector giving which samples belong to which clusters
        counts      k length vector giving the size of each cluster
        temp        k-by-n matrix used as a temporary
        <return>    RMS^2 error, -1.0 for accuracy error, and -2.0 for Python exception
    """
    cdef intp n = data.shape[0], d = data.shape[1], i, j
    cdef bint converged = False
    cdef double S, max_sum, rms2, prevRms2
    cdef double[:,::1] means_orig = means, means_next_orig = means_next, means_tmp

    # Compute initial means
    cdef intp coarseN = (n+KMEANS_ML_RATE//2)//KMEANS_ML_RATE
    if coarseN < KMEANS_ML_MIN_N or coarseN < k:
        # Pick random points for means
        random_subset(data, k, means)
    # Recurse on random subsample to get means - O(coarseN) allocation
    elif __kmeansML(k, random_subset(data, coarseN), means, means_next, membership, counts, temp, high_acc) == -1.0:
        return -1.0 # error - retry with higher accuracy
    
    # Iterate
    with nogil:
        rms2 = km_update(data, means, means_next, membership, counts, temp, high_acc)
        if rms2 == -1.0: return -1.0 # error - retry with higher accuracy
        for _ in xrange(KMEANS_ML_MAX_ITER - 1):
            # Save last state
            prevRms2 = rms2
            means_tmp = means_next; means_next = means; means = means_tmp

            # Compute cluster membership, RMS^2 error, new means, and cluster counts
            rms2 = km_update(data, means, means_next, membership, counts, temp, high_acc)
            if rms2 == -1.0 or rms2 > prevRms2:
                if not high_acc: return -1.0 # error - retry with higher accuracy
                with gil:
                    from warnings import warn
                    err = ('rms^2 had negative components even when using higher accuracy' if rms2 == -1.0 else
                           'rms^2 increased by more than 0.5%% or at the top level even when using high accuracy: %f > %f' % (rms2, prevRms2))
                    warn(err, ClusteringWarning)
                    return -1.0
            else:
                # Check for convergence
                IF KMEANS_ML_ETOL==0.0:
                    converged = prevRms2 == rms2 # actually <= but the < is checked above
                ELSE:
                    # 2*(rmsPrev-rms)/(rmsPrev+rms) <= etol
                    converged = sqrt(prevRms2/rms2) <= (2 + KMEANS_ML_ETOL) / (2 - KMEANS_ML_ETOL)
            if converged:
                max_sum = 0.0
                for i in xrange(k):
                    S = 0.0
                    for j in xrange(d): S += (means[i,j] - means_next[i,j]) * (means[i,j] - means_next[i,j])
                    if S > max_sum: max_sum = S
                if max_sum <= KMEANS_ML_DTOL*KMEANS_ML_DTOL: break

    # At this point the means data is in means_next
    # We need to make sure that refers to the means_orig array or copy the data over
    if &means_orig[0,0] != &means_next[0,0]: means_orig[:,:] = means_next[:,:]

    return rms2

cdef double[:,::1] random_subset(double[:,::1] data, Py_ssize_t n, double[:,::1] out = None):
    """Takes a random n rows from the data array."""
    cdef intp total = data.shape[0], m = data.shape[1], i
    assert(total >= n)
    if out is None: out = PyArray_EMPTY(2, [n, m], NPY_DOUBLE, False)
    cdef intp[::1] inds = PyArray_Arange(0, total, 1, NPY_INTP)
    shuffle_partial(inds, n) # NOTE: shuffles n elements at the END of the array
    for i in xrange(n): out[i,:] = data[inds[total-n+i],:]
    return out

from libc.math cimport isnan
    
cdef double km_update(double[:,::1] data, double[:,::1] means, double[:,::1] means_next,
                      intp[:] membership, intp[::1] counts, double[:,::1] dist, bint high_acc=False) nogil except -2.0:
    """
    K-Means updating. Combines the original private functions computeMembership and computeMeans.
    Now returns RMS^2 instead of RMS.

    Inputs:
        data        n-by-d C matrix, sample features
        means       k-by-d C matrix, current means
        high_acc    if True then use distSqr_acc which is slower but higher accuracy (default False)

    Outputs:
        means_next  k-by-d C matrix, updated means
        membership  n length vector, filled in with which samples belong to which clusters
        counts      k length vector, filled in with size of each cluster
        dist        k-by-n matrix used as a temporary
        <return>    RMS^2 error or -1.0 if accuracy error

    Where:
        k           number of clusters
        d           number of features per sample
        n           number of samples
    """

    # CHANGED: returns the RMS^2 instead of RMS
    cdef intp n = data.shape[0], d = data.shape[1], k = means.shape[0], i, j, p, ix
    cdef double x, min_sum = 0.0
    cdef bint has_empty = False

    # Compute cluster membership and RMS error
    if high_acc: distSqr_acc(data, means, dist) # more accurate but slower, use as necessary
    else:        distSqr(data, means, dist)
    cdef double[::1] mins = dist[0,:] # first row of z starts out as mins
    membership[:] = 0 # all mins are in row 0
    for j in xrange(1, k): # go through all other rows and check for minimums
        for i in xrange(n):
            if dist[j,i] < mins[i]: mins[i] = dist[j,i]; membership[i] = j
    for i in xrange(n):
        if mins[i] < 0: return -1.0 # found a negative value which means we need to increase accurac
        min_sum += mins[i]

    # Compute cluster counts and new means (note: new means are not scaled until later)
    counts[:] = 0
    means_next[:,:] = 0.0
    for i in xrange(n):
        j = membership[i]
        counts[j] += 1
        for p in xrange(d): means_next[j,p] += data[i,p]
    
    # Look for any empty clusters and move their means
    for j in xrange(k):
        if counts[j] == 0:
            # Found an empty cluster
            if not high_acc: return -1.0 # switch to higher accuracy mode if not already high accuracy
            has_empty = True
            
            # Pick the point that is furthest from its clusters mean
            # NOTE: since we have eliminated the first row of dist this won't be able to steal a
            # point from the first cluster.
            x = 0.0
            ix = -1
            for i in xrange(n):
                if counts[membership[i]] > 1 and dist[membership[i],i] > x:
                    x = dist[membership[i],i]
                    ix = i
            i = ix
            if i == -1:
                # No clusters with >1 points found, just pick a random one?
                i = rand() % n
                while counts[membership[i]] == 1: i = rand() % n
            
            # Copy the found point as the new cluster mean
            for p in xrange(d): means[j,p] = data[i,p]
            
            # Adjust memberships and counts
            counts[membership[i]] -= 1
            membership[i] = j
            counts[j] += 1
    # If there were any empty clusters, run km_update again with new means
    # TODO: don't need to do this if we just use the dists already calculated
    if has_empty:
        with gil:
            from warnings import warn
            warn('An empty cluster was found so attempting to adjust for it (this may be an indication that the k is too large)', ClusteringWarning)
        return km_update(data, means, means_next, membership, counts, dist, high_acc)
        
    # Scale new means
    for j in xrange(k):
        x = 1.0 / counts[j]
        for p in xrange(d): means_next[j,p] *= x

    # Return RMS^2 error
    return min_sum / n

cdef void distSqr(double[:,::1] x, double[:,::1] y, double[:,::1] z) nogil:
    """
    Return matrix of all-pairs squared distances between the vectors in the columns of x and y.
    
    Equivilent to:
        # Slow method:
        return ((x[None,:,:]-y[:,None,:])**2).sum(2)

        # Fast method:
        z = x.dot(y.T).T
        z *= -2
        z += (x*x).sum(1)[None,:]
        z += (y*y).sum(1)[:,None]
        return z

    INPUTS
        x       n-by-k C matrix
        y       m-by-k C matrix

    OUTPUT
        z       m-by-n C matrix

    This is an optimized version written in Cython. It works just like the original but is faster
    and uses less memory. The matrix multiplication is done with BLAS (using dgemm to be able to do
    the multiply and addition all at the same time). The squares and sums are done with looping.
    The original uses O((n+m)*(k+1)) bytes of memory while this allocates no memory (besides what
    dgemm uses internally).

    Note: this particular function uses the "fast" method listed above and has the tendency to
    introduce numerical errors during calculation. Ultimately, under certain circumstances, this
    will in fact return negative values (even though they should be squared distances)! If these
    numerical issues are okay, then all negative values should just be simply set to 0 and ignored.
    If not then the distSqr_acc function should be used instead.
    """
    # NOTE: this is a near-commutative operation, e.g. distSqr(x, y, z) is the same as distSqr(y, x, z.T)
    cdef int n = x.shape[0], m = y.shape[0], k = x.shape[1], i, j, p
    cdef double alpha = -2.0, beta = 1.0, sum2
    cdef double[:] temp = z[m-1,:] # last row in z is used as a temporary
    # temp = (x*x).sum(1)
    for i in xrange(n):
        sum2 = 0.0
        for p in xrange(k): sum2 += x[i,p]*x[i,p]
        temp[i] = sum2
    # z = temp[None,:] + (y*y).sum(1)[:,None]
    for j in xrange(m):
        sum2 = 0.0
        for p in xrange(k): sum2 += y[j,p]*y[j,p]
        for i in xrange(n): z[j,i] = sum2 + temp[i]
    # z += -2*x.dot(y.T)
    dgemm("T", "N", &n, &m, &k, &alpha, &x[0,0], &k, &y[0,0], &k, &beta, &z[0,0], <int*>&z.shape[1])
   
cdef void distSqr_acc(double[:,::1] x, double[:,::1] y, double[:,::1] z) nogil:
    """
    This is just like distSqr except that it is has greatly increased accuracy at the cost of
    speed. These differences are due to performing the subtraction early instead of late. This uses
    the concept of the "slow method" decribed in the documentation for distSqr.
                    
    This still does not allocate any extra memory (and doesn't use dgemm so no hidden memory
    allocation may happen there). This is about 3x to 10x slower than distSqr.
    """
    cdef intp n = x.shape[0], m = y.shape[0], k = x.shape[1], i, j
    for i in prange(m):
        for j in xrange(n): z[i,j] = distSqr_vector(x[j], y[i], k)

cdef inline double distSqr_vector(double[::1] x, double[::1] y, intp k) nogil:
    """Squared distance between two k-length vectors x and y."""
    cdef intp p
    cdef double sum2 = 0.0
    for p in xrange(k): sum2 += (x[p]-y[p])*(x[p]-y[p])
    return sum2


#################### Gradient Descent ####################

# 1/(1+exp(-37)) produces 1.0 and 1/(1+exp(709)) is 1.217e-308 (close enough to 0)
# outside of this range we start getting nans, infs, and Numpy errors
DEF MIN_S_VALUE = -709.0
DEF MAX_S_VALUE = 37.0
min_s_value = MIN_S_VALUE
max_s_value = MAX_S_VALUE

# During the equation c1/(1-g_i)*g_i we want to make sure g_i in the denominator is
# not exactly 1.0 or we get division-by-zero errors so we use this value instead.
DEF ALMOST_ONE = 1.0 - 1e-16

def gradient(double[::1] f, double[:,::1] g, double[:,:,::1] s, double[::1,:] x, double[::1] y, double[:,:,::1] grads):
    """
    Calculates the gradient of the error using equation 12/13 from Seyedhosseini et al 2013 for a batch
    of samples. The summed output across all samples in the batch is saved to the grads matrix.
    """
    grads[:,:,:] = 0.0
    cdef double c1, c2, c3
    cdef intp N = s.shape[0], M = s.shape[1], P = s.shape[2], n = x.shape[0], i, j, k, p
    #with nogil: # TODO: decide if with nogil has any impact on single-threaded performance here
    for p in xrange(P):
        c1 = -2.0*(y[p]-f[p])*(1.0-f[p])
        for i in xrange(N):
            c2 = c1/(1.0-min(g[i,p], ALMOST_ONE))*g[i,p]
            for j in xrange(M):
                c3 = c2*(1.0-s[i,j,p])
                for k in xrange(n):
                    grads[i,j,k] += c3*x[k,p]

def descent(double[:,:,::1] grads, double[:,:,::1] prevs, double[:,:,::1] W, double rate, double momentum):
    """
    Updates the weights based on gradient descent using the given learning rate and momentum. The grads
    are the summed values instead of the averages. The rate and momentum values should be per-sample. 
    """
    cdef intp N = grads.shape[0], M = grads.shape[1], n = grads.shape[2], i, j, k
    #with nogil: # TODO: decide if with nogil has any impact on single-threaded performance here
    for i in xrange(N):
        for j in xrange(M):
            for k in xrange(n):
                prevs[i,j,k] = grads[i,j,k] + momentum*prevs[i,j,k]
                W[i,j,k] -= rate*prevs[i,j,k]

def descent_do(double[:,:,::1] grads, double[:,:,::1] prevs, double[:,:,::1] W,
               intp[::1] i_order, intp[::1] j_order, double rate, double momentum):
    """
    Updates the weights based on gradient descent with dropout using the given learning rate (which should
    be scaled based on the batch size) and momentum. The matrices prevs and W represent the entire dataset
    and the i_order and j_order arrays which regions in those matrices to use. The grads matrix represents
    just the non-dropped out data.
    """
    cdef intp Nd = grads.shape[0], Md = grads.shape[1], n = grads.shape[2], i, j, k
    cdef dbl_p W_ij, p_ij
    #with nogil: # TODO: decide if with nogil has any impact on single-threaded performance here
    for i in xrange(Nd):
        for j in xrange(Md):
            p_ij = &prevs[i_order[i],j_order[j],0]
            W_ij = &W[i_order[i],j_order[j],0]
            for k in xrange(n):
                p_ij[k] = grads[i,j,k] + momentum*p_ij[k]
                W_ij[k] -= rate*p_ij[k]

def gradient_descent(double[:,:] X, char[::1] Y, double[:,:,::1] W,
                     const intp niters, const double rate, const double momentum, target, disp=None):
    """
    This is an optimized version of gradient descent that always has dropout=False and batchsz=1. See
    chm.ldnn.gradient_descent for more information about the other parameters.
    
    ?
    Slightly faster when X is Fortran-ordered but not by much (possibly about 15%). However that would mean
    that the entire X array, which is normally C-ordered, would have to be copied and stored in memory and
    it is huge.
    """

    # Matrix sizes
    cdef intp N = W.shape[0], M = W.shape[1], n = W.shape[2], P = Y.shape[0]

    # Allocate memory
    cdef intp[::1] order = PyArray_Arange(0, P, 1, NPY_INTP)
    cdef double[:,:,::1] prevs = PyArray_ZEROS(3, [N,M,n], NPY_DOUBLE, False)
    cdef double[:,::1] s = PyArray_ZEROS(2, [N,M], NPY_DOUBLE, False)
    cdef double[::1]   g = PyArray_ZEROS(1, &N,    NPY_DOUBLE, False)
    cdef double[::1]   x = PyArray_EMPTY(1, &n,    NPY_DOUBLE, False) # a single row from X
    cdef ndarray total_error = PyArray_EMPTY(1, &niters, NPY_DOUBLE, False)
    
    # Variables
    cdef intp i, p
    cdef double totalerror, y, lower_target, upper_target, target_diff
    lower_target,upper_target = target
    target_diff = upper_target-lower_target
    
    for i in xrange(niters):
        with nogil:
            totalerror = 0.0
            shuffle(order)
            for p in xrange(P):
                x[:] = X[:,order[p]] # copying here greatly increases overall speed 
                y = lower_target + target_diff*<bint>Y[order[p]]
                totalerror += _grad_desc(x, y, W, prevs, s, g, rate, momentum)
        total_error[i] = sqrt(totalerror/P)
        if disp is not None: disp('Iteration #%d error: %f' % (i+1,total_error[i]))
    return total_error

cdef double _grad_desc(const double[::1] x, const double y, double[:,:,::1] W, double[:,:,::1] prevs,
                       double[:,::1] s, double[::1] g, const double rate, const double momentum) nogil:
    """A single step and sample for gradient descent."""
    cdef intp N = W.shape[0], M = W.shape[1], n = W.shape[2], i, j, k
    
    # Calculate the sigmas, gs, and classifier (eqs 9 and 10 from Seyedhosseini et al 2013) 
    cdef double f = 1.0, g_i, s_ij
    for i in xrange(N):
        g_i = 1.0
        for j in xrange(M):
            s_ij = 0.0
            for k in xrange(n): s_ij += W[i,j,k]*x[k]
            s_ij = clip(s_ij, MIN_S_VALUE, MAX_S_VALUE)
            s_ij = 1.0/(1.0+exp(-s_ij))
            s[i,j] = s_ij
            g_i *= s_ij
        g[i] = g_i
        f *= 1.0-g_i
    f = 1.0-f
    
    # Calculate gradient (eqs 12 and 13 from Seyedhosseini et al 2013)
    # and perform a gradient descent step
    cdef double yf = y-f, c1 = -2.0*(1.0-f)*yf, c2, c3
    for i in xrange(N):
        c2 = c1/(1.0-min(g[i], ALMOST_ONE))*g[i]
        for j in xrange(M):
            c3 = c2*(1.0-s[i,j])
            for k in xrange(n):
                prevs[i,j,k] = x[k]*c3 + momentum*prevs[i,j,k]
                W[i,j,k] -= rate*prevs[i,j,k]

    # Calculate error (part of eq 11 from Seyedhosseini et al 2013)
    return yf*yf


def gradient_descent_dropout(double[:,:] X, char[::1] Y, double[:,:,::1] W,
                             const intp niters, const double rate, const double momentum, target, disp=None):
    """
    This is an optimized version of gradient descent that always has dropout=True and batchsz=1. See
    chm.ldnn.gradient_descent for more information about the other parameters.
    
    Slightly faster when X is Fortran-ordered but not by much (possibly about 15%). However that
    would mean that the entire X array, which is normally C-ordered, would have to be copied and
    stored in memory and it is huge.
    """

    # Matrix sizes
    cdef intp N = W.shape[0], M = W.shape[1], n = W.shape[2], P = Y.shape[0], N2 = N//2, M2 = M//2

    # Allocate memory
    cdef intp[::1] order   = PyArray_Arange(0, P, 1, NPY_INTP)
    cdef intp[::1] i_order = PyArray_Arange(0, N, 1, NPY_INTP)
    cdef intp[::1] j_order = PyArray_Arange(0, M, 1, NPY_INTP)
    cdef double[:,:,::1] prevs = PyArray_ZEROS(3, [N,M,n], NPY_DOUBLE, False)
    cdef double[:,::1] s = PyArray_ZEROS(2, [N2,M2], NPY_DOUBLE, False) # stores shuffled data
    cdef double[::1]   g = PyArray_ZEROS(1, &N2,     NPY_DOUBLE, False) # stores shuffled data
    cdef double[::1]   x = PyArray_EMPTY(1, &n,      NPY_DOUBLE, False) # a single row from X
    cdef ndarray total_error = PyArray_EMPTY(1, &niters, NPY_DOUBLE, False)
    
    # Variables
    cdef intp i, p
    cdef double totalerror, y, lower_target, upper_target, target_diff
    lower_target,upper_target = target
    target_diff = upper_target-lower_target
    
    for i in xrange(niters):
        with nogil:
            totalerror = 0.0
            shuffle(order)
            for p in xrange(P):
                shuffle(i_order)
                shuffle(j_order)
                x[:] = X[:,order[p]] # copying here greatly increases overall speed 
                y = lower_target + target_diff*<bint>Y[order[p]]
                totalerror += _grad_desc_do(x, y, W, prevs, i_order[:N2], j_order[:M2], s, g, rate, momentum)
        total_error[i] = sqrt(totalerror/P)
        if disp is not None: disp('Iteration #%d error: %f' % (i+1,total_error[i]))
    return total_error

cdef double _grad_desc_do(const double[::1] x, const double y, double[:,:,::1] W, double[:,:,::1] prevs,
                          const intp[::1] i_order, const intp[::1] j_order, double[:,::1] s, double[::1] g,
                          const double rate, const double momentum) nogil:
    """A single step and sample for gradient descent dropout."""
    cdef intp N2 = i_order.shape[0], M2 = j_order.shape[0], n = x.shape[0], i, j, k
    
    # Calculate the sigmas, gs, and classifier (eqs 9 and 10 from Seyedhosseini et al 2013) 
    cdef double f = 1.0, g_i, s_ij
    cdef dbl_p W_ij, p_ij
    for i in xrange(N2):
        g_i = 1.0
        for j in xrange(M2):
            s_ij = 0.0
            W_ij = &W[i_order[i],j_order[j],0]
            for k in xrange(n): s_ij += W_ij[k]*x[k]
            s_ij = clip(s_ij, MIN_S_VALUE, MAX_S_VALUE)
            s_ij = 1.0/(1.0+exp(-s_ij))
            s[i,j] = s_ij
            g_i *= s_ij
        g[i] = g_i
        f *= 1.0-g_i
    f = 1.0-f
    
    # Calculate gradient (eqs 12 and 13 from Seyedhosseini et al 2013)
    # and perform a gradient descent step
    cdef double yf = y-f, c1 = -2.0*(1.0-f)*yf, c2, c3
    for i in xrange(N2):
        c2 = c1/(1.0-min(g[i], ALMOST_ONE))*g[i]
        for j in xrange(M2):
            p_ij = &prevs[i_order[i],j_order[j],0]
            W_ij = &W[i_order[i],j_order[j],0]
            c3 = c2*(1.0-s[i,j])
            for k in xrange(n):
                p_ij[k] = x[k]*c3 + momentum*p_ij[k]
                W_ij[k] -= rate*p_ij[k]

    # Calculate error (part of eq 11 from Seyedhosseini et al 2013)
    return yf*yf
