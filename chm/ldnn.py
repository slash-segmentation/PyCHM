"""
Logistic Disjunctive Normal Network (LDNN) Model implementation.

This implements the algorithm as described in Seyedhosseini, Sajjadi, and Tasdizen (2013).

This was originally in the MATLAB files LearnAndOrNetMEX, UpdateDiscriminants(_SB),
EvaluateAndOrNetMEX, and genOutput(_SB). Most variables and functions have been renamed to match
the paper, e.g. discriminants changed to w or weights. Some parts were removed, such as validation
and regularization from LearnAndOrNetMEX (most of it had already been removed).

As a shortcut the biases b_ij are lumped into the weights w_ijk and an extra "feature" is added to
the feature vector X that is always 1.

Several of these methods were in C code before but have been converted into Python because it is
either faster or no significant difference. Several are in the Cython module __ldnn.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

# TODO: make most of this its own package separate from CHM (already being used by GLIA)

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from .classifier import Classifier

class LDNN(Classifier):
    """
    LDNN classifier from Seyedhosseini, Sajjadi, and Tasdizen - 2013. It uses a stochastic gradient
    descent with either mini-batches or the "dropout" method.
    
    There are several parameters that influence the classifier:
        N           number of groups (number of ORs)
        M           number of nodes/discriminants per group (number of ANDs)
        downsample  amount of downsampling to perform on the data for clustering
        kmeans_rep  times to repeat running k-means clustering looking for a better solution
        dropout     if True then perform input dropout at 50% according to Hinton et al. 2012
        batchsz     if not using dropout then this is the size of the batches to process
        niters      number of iterations of gradient descent to run
        rate        the step size or learning rate of gradient descent
        target      target values to prevent oversatuation of sigmoid function
                    see LeCun, Bottou, Orr, Muller 1998 section 4.5
        momentum    the momentum of the learning process of gradient descent
    """
    __weights = None # combined w_ijk and b_ij values, shaped like NxMx(n+1)
    __params = None
    
    _def_params_L0S1 = {'N':10,'M':20,'downsample':10,'kmeans_rep':1,
                        'dropout':False,'batchsz':100,'niters':15,'rate':0.005,'target':(0.1,0.9),'momentum':0.5}
    _def_params_L0   = {'N':10,'M':20,'downsample':10,'kmeans_rep':1,
                        'dropout':False,'batchsz':100,'niters':6, 'rate':0.005,'target':(0.1,0.9),'momentum':0.5}
    _def_params      = {'N':24,'M':24,'downsample':10,'kmeans_rep':1,
                        'dropout':True, 'batchsz':1,  'niters':15,'rate':0.025,'target':(0.1,0.9),'momentum':0.5}
    
    @classmethod
    def get_default_params(cls, stage, level):
        if level == 0: return cls._def_params_L0S1 if stage == 1 else cls._def_params_L0
        return cls._def_params
    
    def __init__(self, params=None, weights=None):
        """
        Creates or loads an LDNN classifier.
        
        The parameters for the classifer are specified as a dictionary. See the class documentation
        for available parameters.
        
        If the model is already learned then the weights must be provided. They must be a NxMxn array
        (where n can be any value).
        """
        from numpy import float64, ndarray
        self.__params = LDNN._def_params.copy()
        if params is not None: self.__params.update(params)
        if weights is not None:
            assert(weights.shape[:2] == (self.__params['N'],self.__params['M']))
            self.__weights = weights.view(ndarray).astype(float64, copy=False)

    def print(self, prnt=print):
        super(LDNN, self).print(prnt)
        prnt("Parameters:")
        for item in sorted(self.__params.iteritems()): prnt("  %s: %s"%item)
        if self.__weights is not None:
            prnt("Weights:")
            N,M,_ = self.__weights.shape
            for i in xrange(N):
                for j in xrange(M):
                    prnt("  %2d-%2d: %s"%(i,j,' '.join(str(x) for x in self.__weights[i][j])))

    @property
    def params(self): return self.__params.copy()
    @property
    def learned(self): return self.__weights is not None
    @property
    def features(self): return self.__weights.shape[-1] if self.learned else None
    @property
    def extra_features(self): return 1
    def copy(self): return self.__class__(self.__params)
    @property
    def evaluation_memory(self):
        from numpy import float64
        N,M,n,itmsz = self.__params['N'], self.__params['M'], self.features or 0, float64(0).itemsize
        return itmsz*(n+N*(M+1))
    
    def evaluate(self, X, nthreads=1):
        assert(self.__weights is not None)
        return test(self.__weights, X, self.__params.get('dropout', False), nthreads)
    
    def learn(self, X, Y, nthreads=1):
        assert(self.__weights is None)
        self.__weights = learn(X, Y, nthreads=nthreads, **self.__params)

def f_(s, dropout=False, out=None):
    """
    Calculate f = 1-prod(1-prod(sigma_ij)) from equation 10 in Seyedhosseini et al 2013.
    
    Inputs:
        s       NxMxS matrix of sigma_ij values where N is the number of groups (ORs), M is the
                number of nodes/discriminants per group (ANDs), and S is the number of samples
        dropout Calculate 1-sqrt(prod(1-sqrt(prod(sigma_ij)))) [default is False]
    
    Returns a contiguous length-P array of the classifier results.
    """
    from numpy import rollaxis, prod, sqrt, subtract
    g = prod(rollaxis(s, 2), axis=2)
    del s
    f = prod(subtract(1.0, sqrt(g, g) if dropout else g, out=g), axis=1, out=out)
    del g
    return subtract(1.0, sqrt(f, f) if dropout else f, out=f)

def f_g(s, out_f=None, out_g=None):
    """
    Calculate g_i=prod(sigma_ij) and f=1-prod(1-g_i) from equation 10 in Seyedhosseini et al 2013.
    
    Inputs:
        s    NxMxS matrix of sigma_ij values where N is the number of groups (ORs), M is the number
             of nodes/discriminants per group (ANDs), and S is the number of samples
    
    Returns a contiguous length-P array of the classifier results (f) and a C-contiguous NxP matrix
    of the intermediate g_i values.
    """
    from numpy import rollaxis, prod, subtract
    if out_g is not None: out_g = out_g.T
    g = prod(rollaxis(s, 2), axis=2, out=out_g)
    del s
    f = prod(subtract(1.0, g), axis=1, out=out_f)
    return (subtract(1.0, f, out=f), g.T)

def sigma_ij(W, X, out=None):
    """
    Calculates sigma_ij = 1/(1+exp(-W*X+B))) from equation 9 in Seyedhosseini et al 2013.
    
    Inputs:
        W   NxMx(n+1) matrix of weights representing all w_ijk and biases b_ij (in the final row).
            n is the number of features, N is the number of groups (ORs), and M is the of
            nodes/discriminants per group (ANDs)
        X   nxS or (n+1)xS matrix of samples where S is the number of samples. If it is (n+1)xS then
            the last row will be filled with 1s by this method. Using an (n+1)xS input is faster.
    
    Outputs:
        out NxMxP matrix of the sigma_ij values

    It is best if W, X, and out are contiguous, particularily C-contiguous.
    """
    # Some notes on "dot":
    #   regardless of inputs always produces C-contiguous output
    #   fastest when both inputs are C ordered and ~10% slower when both inputs are F ordered
    #   mixed inputs (one input is C and the other is F) lie inbetween
    from numpy import clip, exp, negative, divide, add
    from .__ldnn import min_s_value, max_s_value #pylint: disable=no-name-in-module
    N,M,n = W.shape # the n is actually n+1
    n_,S = X.shape
    if out is not None:
        if out.shape != (N,M,S): raise ValueError()
        out = out.reshape(N*M, S)
    W = W.reshape(N*M, n)
    if n_ == n:
        X[-1].fill(1)
        s = W.dot(X, out=out)
    elif n_ == n - 1:
        s = W[:,:-1].dot(X, out=out)
        s += W[:,-1:]
    else: raise ValueError()
    del W,X
    clip(s, min_s_value, max_s_value, out=s)
    return divide(1.0, add(1.0, exp(negative(s, out=s), out=s), out=s), out=s).reshape(N, M, S)

def __print(s, depth=2):
    """
    Like print(...) but pre-pends the current timestamp, spaces dependent on the depth, and forces
    a flush.
    """
    import sys, datetime
    print('%s %s%s'%(str(datetime.datetime.utcnow())[:19], '  '*depth, s))
    sys.stdout.flush()

def test(W, X, dropout=False, nthreads=1):
    """
    Perform LDNN testing. This just uses sigma_ij and f_ functions along with calling
    chm.utils.set_lib_threads(nthreads) to set the number of threads to use.
    
    Inputs:
        W   NxMx(n+1) matrix of weights representing all w_ijk and biases b_ij (in the final row).
            n is the number of features, N is the number of groups (ORs), and M is the of
            nodes/discriminants per group (ANDs)
        X   nxS or (n+1)xS matrix of samples where S is the number of samples. If it is (n+1)xS then
            the last row will be filled with 1s by this method. Using an (n+1)xS input is faster.
        dropout if the weights were calculated using dropout [default is False]

    Returns a contiguous length-P array of the classifier results.

    It is best if W and X are contiguous, particularily C-contiguous.
    """
    from .utils import set_lib_threads
    set_lib_threads(nthreads)
    return f_(sigma_ij(W, X), dropout)

def filter_clustering_warnings(action='always'):
    """
    Set the filter action for ClusteringWarnings generated by K-means when there is an empty
    cluster. The action can be one of 'error', 'ignore', 'always', 'default', 'module', or 'once'.
    """
    #pylint: disable=no-member
    from . import __ldnn
    import warnings
    warnings.simplefilter(action, __ldnn.ClusteringWarning)

def learn(X, Y, N=5, M=5, downsample=1, kmeans_rep=5,  #pylint: disable=too-many-arguments
          dropout=False, niters=25, batchsz=1, rate=0.025, momentum=0.5, target=(0.1,0.9), disp=True, nthreads=1):
    """
    Calculate the initial weights using multilevel K-Means clustering on a subset of the data. The
    clusters are used according to the end of section 3 of Seyedhosseini et al 2013.
    
    Inputs:
        X   (n+1)xS matrix of feature vectors where n is the number of features and S is the number
            of samples, the last row will be filled with 1s by this method, must be Fortran-ordered
        Y   S-length array of bool labels
    
    Parameters:
        N           number of groups to create (ORs)
        M           number of nodes/discriminants per group to create (ANDs)
        downsample  amount of downsampling to perform on the data for clustering
        kmeans_rep  times to repeat running k-means clustering looking for a better solution
        dropout     if True then perform input dropout at 50% according to Hinton et al. 2012
        niters      number of times to go through all of the samples during gradient descent
        batchsz     size of the mini-batches in gradient descent if dropout is False
        rate        gradient descent learning rate
        momentum    amount the previous gradient influences the next gradient
        target      target values, default to 0.1 and 0.9, see LeCun et al 1998 section 4.5
        disp        display messages about progress
        nthreads    number of threads to use during clustering
    
    Output:
        A C-contiguous NxMx(n+1) matrix of initial weights representing all w_ijk and biases b_ij
        (in the final row).
    """
    from .utils import set_lib_threads
    if disp:
        pos = Y.sum()
        __print('Number of training samples = %d (%d positive, %d negative)'%(X.shape[1],pos,X.shape[1]-pos))
        __print('Calculating initial weights...')
    W = init_weights(X, Y, N, M, downsample, kmeans_rep, True, nthreads)
    if disp:
        set_lib_threads(nthreads)
        __print('Initial error: %f'%calc_error(X,Y,W,target,dropout), 3)
        __print('Gradient descent...')
    set_lib_threads(1) # always use 1 thread during gradient descent
    gradient_descent(X, Y, W, niters, batchsz, dropout, rate, momentum, target, disp)
    set_lib_threads(nthreads)
    if disp:
        __print('Final error: %f'%calc_error(X,Y,W,target,dropout), 3)
    return W

def init_weights(X, Y, N=5, M=5, downsample=1, kmeans_rep=5, whiten=False, nthreads=1):
    """
    Calculate the initial weights using multilevel K-Means clustering on a subset of the data. The
    clusters are used according to the end of section 3 of Seyedhosseini et al 2013.
    
    Inputs:
        X   (n+1)xS matrix of feature vectors where n is the number of features and S is the number
            of samples, the last row will be filled with 1s by this method
        Y   S-length array of bool labels
        
    Parameters:
        N   number of groups to create (ORs)
        M   number of nodes/discriminants per group to create (ANDs)
        downsample  amount of downsampling to perform on the data before clustering
        kmeans_rep  times to repeat running k-means clustering looking for a better solution
        whiten      scale each feature vector in X to have a variance of 1 before running k-means
        nthreads    the number of threads to use (will not use more than 2 threads though)

    Output:
        A C-contiguous NxMx(n+1) matrix of initial weights representing all w_ijk and biases b_ij
        (in the final row).
    """
    from numpy.linalg import norm
    from . import __ldnn
    from .utils import set_lib_threads
    
    #from numpy.random import randn
    #return randn(N, M, X.shape[0]) #random initialization
    
    # NOTE: the extra 'feature' is set to all 1s at this point and won't effect k-means
    X[-1,:] = 1.0

    # First bullet point on page 5 of Seyedhosseini et al 2013
    # Calculating C+ and C- using k-means
    num_pos = Y.sum()
    num_neg = len(Y) - Y.sum()
    set_lib_threads(max(min(nthreads, num_pos // (500*downsample)), 1))
    Cp = __ldnn.run_kmeans(N, X,  Y, downsample, kmeans_rep, whiten, nthreads) #pylint: disable=no-member
    set_lib_threads(max(min(nthreads, num_neg // (500*downsample)), 1))
    Cn = __ldnn.run_kmeans(M, X, ~Y, downsample, kmeans_rep, whiten, nthreads) #pylint: disable=no-member
    
    # Second and third bullet points on page 5 of Seyedhosseini et al 2013
    # Calculating initial weights and biases
    # Weights are unit-length vectors from each C- and C+ pair
    # Biases are setup so that sigma values are 0.5 at the midpoint between each C- and C+ pair
    weights = Cp[:,None,:] - Cn[None,:,:]
    baises  = Cp[:,None,:] + Cn[None,:,:]
    weights /= norm(weights, axis=2, keepdims=True) # normalize the vectors
    baises *= weights
    baises = baises.sum(axis=2, out=weights[:,:,-1])
    baises *= -0.5
    return weights

def calc_error(X, Y, W, target=(0.1,0.9), dropout=False):
    """
    Calculate the square root of the error of the model (eq 11 from Seyedhosseini et al 2013).
    
    Inputs:
        X   (n+1)xS matrix of feature vectors
        Y   S-length array of bool labels
        W   NxMx(n+1) matrix of weights and biases
        
    Parameters:
        target  target values, default to 0.1 and 0.9, see LeCun et al 1998 section 4.5
        dropout if the weights were calculated using dropout [default is False]
        
    Note that if X/Y/W are the dropped-out subsets than dropout should be kept as False.
    """
    from math import sqrt
    lower_target,upper_target = target
    Y = (upper_target-lower_target)*Y
    Y += lower_target
    Y -= f_(sigma_ij(W, X), dropout)
    Y *= Y
    return sqrt(Y.mean())

def gradient_descent(X, Y, W, niters=25, batchsz=1, dropout=False, rate=0.025, momentum=0.5, target=(0.1,0.9), disp=True): #pylint: disable=too-many-arguments
    """
    Uses stochastic gradient descent with mini-batches and dropout to minimize the weights of
    the LDNN classifier using the training data provided. Parts of this are implemented in Cython.
    
    As a special case when batchsz=1 a pure Cython version is used which is about 10x faster. This
    could mostly be mitigated if special versions of the sigma_ij and f_g functions were made that
    were optimized for single samples.
    
    Inputs:
        X   (n+1)xS matrix of feature vectors where n is the number of features and S is the number
            of samples
        Y   S-length array of bool labels
        
    Inputs/Outputs:
        W   NxMx(n+1) matrix of weights and biases, C-ordered

    Outputs:
        The total error for each epoch is returned as sqrt(sum(error^2)/P)

    Parameters:
        niters      number of times to go through all of the samples
        batchsz     size of the mini-batches
        dropout     if True then perform input dropout at 50% according to Hinton et al. 2012
        rate        influence of gradient during each step, per sample
        momentum    amount the previous gradient influences the next gradient, per sample
        target      target values, default to 0.1 and 0.9, see LeCun et al 1998 section 4.5
        disp        display messages about progress
    """
    #pylint: disable=too-many-locals
    from numpy import arange, zeros, empty, int8
    from math import sqrt
    from . import __ldnn
    from .__shuffle import shuffle #pylint: disable=no-name-in-module

    assert(Y.ndim == 1 and W.ndim == 3 and X.shape == (W.shape[2],Y.shape[0]))
    assert(X.flags.forc and Y.flags.forc and W.flags.c_contiguous)

    dropout = bool(dropout)
    
    if batchsz == 1:
        # Use the fully optimized version for this situation
        f = __ldnn.gradient_descent_dropout if dropout else __ldnn.gradient_descent #pylint: disable=no-member
        disp = (lambda s:__print(s,3)) if disp else None
        return f(X, Y.view(int8), W, niters, rate, momentum, target, disp)
    
    # Matrix sizes
    (N,M,n),S = W.shape,len(Y)
    N2,M2 = (N//2,M//2) if dropout else (N,M)
    
    # Allocate memory
    order = arange(S)
    total_error = empty(niters)
    f_full,g_full = empty(batchsz), empty((N2,batchsz))
    grads,prevs = empty((N2,M2,n)), zeros((N,M,n))

    # Get Cython functions and handle dropout differences
    grad = __ldnn.gradient #pylint: disable=no-member
    if dropout:
        i_order,j_order = arange(N), arange(M)
        cy_desc = __ldnn.descent_do #pylint: disable=no-member
        desc = lambda:cy_desc(grads, prevs, W, i_order, j_order, rate, momentum)
    else:
        cy_desc = __ldnn.descent #pylint: disable=no-member
        desc = lambda:cy_desc(grads, prevs, W, rate, momentum)

    lower_target,upper_target = target
    target_diff = upper_target-lower_target
        
    for it in xrange(niters):
        shuffle(order)
        totalerror = 0.0
        f,g = f_full,g_full
        for s in xrange(0, S, batchsz):
            x = X[:,order[s:s+batchsz]] # x is always F-ordered even if X is C-ordered
            y = target_diff*Y[order[s:s+batchsz]]; y+=lower_target # see LeCun, Bottou, Orr, Muller 1998 section 4.5

            # Calculate the sigmas, gs, and classifier (eqs 9 and 10 from Seyedhosseini et al 2013) 
            if dropout:
                shuffle(i_order)
                shuffle(j_order)
                W_ = W[i_order[:N2,None],j_order[:M2],:]
            else: W_ = W
            sij = sigma_ij(W_, x)
            if len(y) != batchsz: f,g = f_g(sij) # the last mini-batch might not be full
            else: f_g(sij, f, g)

            # Calculate error (part of eq 11 from Seyedhosseini et al 2013)
            yf2 = y-f; yf2 *= yf2
            totalerror += yf2.sum()
            
            # Calculate gradient (eqs 12 and 13 from Seyedhosseini et al 2013)
            grad(f, g, sij, x, y, grads) # implemented in Cython
            
            # Update weights with a gradient step across the minibatch
            desc() # implemented in Cython
            
        # Report total error for the iteration
        total_error[it] = sqrt(totalerror/S) # Similar to eq 11 from Seyedhosseini et al 2013 but slightly different
        if disp: __print('Iteration #%d error: %f' % (it+1,total_error[it]), 3)
    return total_error
