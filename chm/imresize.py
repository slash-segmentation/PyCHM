"""
imresize function that mostly emulates MATLAB's imresize function. Additionally a 'fast' variant is
provided that always halves the image size with bicubic and antialiasing.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from pysegtools.general.delayed import delayed
from numpy import finfo, float64

__all__ = ('imresize', 'imresize_fast')

__methods = delayed(lambda:{
    'nearest' : (box,1), 'bilinear' : (triangle,2), 'bicubic' : (cubic,4),
    'box'     : (box,1), 'triangle' : (triangle,2), 'cubic'   : (cubic,4),
    'lanczos2' : (lanczos2,4), 'lanczos3' : (lanczos3,6),
}, dict)
__eps = delayed(lambda:finfo(float64).eps, float64)


def imresize(im, scale_or_output_shape, method='bicubic', antialiasing=None, out=None, nthreads=1):
    """
    Resize an image.

    scale_or_output_shape is one of:
        floating-point number for scale
        tuple/list of 2 floating-point numbers for multi-scales
        tuple/list of 2 ints for output_shape (supporting None for calculated dims)

    method is one of:
        'nearest' or 'box'
        'bilinear' or 'triangle'
        'bicubic' or 'cubic' (default)
        'lanczos2'
        'lanczos3'
        tuple/list of a kernel function and a kernel width

    antialiasing defaults to True except for nearest which is False (box is True as well)

    Unlike MATLAB's imresize, the following features are not supported:
      * gpu-arrays
      * indexed images (so none of the params map, Colormap, Dither), however this can be
        accomplished outside of the function
      * 0 or 1 dimensional images, however this can be accomplished with adding length-1 dimensions
        outside the function
    """
    from numpy import greater
    
    # Parse arguments scale_or_output_shape, method, and antialiasing
    sh = im.shape
    scale, out_shape = __scale_shape(sh, scale_or_output_shape)
    antialiasing = method != 'nearest' if antialiasing is None else bool(antialiasing)
    kernel, kernel_width = __methods.get(method, method)

    # Handle the im and out arguments
    im, out, dt = __im_and_out(im, out, out_shape)
    
    # Calculate interpolation weights and indices for each dimension.
    wghts1, inds1 = __contributions(sh[0], out.shape[0], scale[0], kernel, kernel_width, antialiasing)
    if sh[0] == sh[1] and out.shape[0] == out.shape[1] and scale[0] == scale[1]:
        wghts2, inds2 = wghts1, inds1
    else:
        wghts2, inds2 = __contributions(sh[1], out.shape[1], scale[1], kernel, kernel_width, antialiasing)

    # Resize the image
    __imresize(im, out, (wghts1, wghts2), (inds1, inds2), scale, nthreads)
    
    # Return the output array (possibly converted back to logicals)
    return greater(out, 128, out.view(dt)) if dt.kind == 'b' else out

def __imresize(im, out, wghts, inds, scale, nthreads):
    from numpy import empty, ndindex
    if wghts[0] is None and wghts[1] is None:
        return im.take(inds[0], 0).take(inds[1].T, 1, out)
    sh = im.shape
    imr, t_sh = ((__imresize_01, (out.shape[0], sh[1])) if scale[0] <= scale[1] else
                 (__imresize_10, (sh[0], out.shape[1])))
    tmp = empty(t_sh, im.dtype, order='F' if im.flags.fortran else 'C')
    if len(sh) == 2: imr(im, tmp, out, wghts, inds, nthreads)
    else:
        base = slice(None), slice(None)
        for idx in ndindex(sh[2:]): imr(im[base+idx], tmp, out[base+idx], wghts, inds, nthreads)
    return out

def __imresize_01(im, tmp, out, weights, indices, nthreads):
    from .__imresize import imresize #pylint: disable=redefined-outer-name, no-name-in-module
    if weights[0] is None: im.take(indices[0], 0, tmp) # nearest neighbor
    else: imresize(im, tmp, weights[0], indices[0], nthreads)
    if weights[1] is None: tmp.T.take(indices[1], 1, out.T) # nearest neighbor
    else: imresize(tmp.T, out.T, weights[1], indices[1], nthreads)

def __imresize_10(im, tmp, out, weights, indices, nthreads):
    from .__imresize import imresize #pylint: disable=redefined-outer-name, no-name-in-module
    if weights[1] is None: im.T.take(indices[1], 1, tmp.T) # nearest neighbor
    else: imresize(im.T, tmp.T, weights[1], indices[1], nthreads)
    if weights[0] is None: tmp.take(indices[0], 0, out) # nearest neighbor
    else: imresize(tmp, out, weights[0], indices[0], nthreads)

def __scale_shape(sh, scale_or_shape):
    from math import ceil
    from numbers import Real, Integral
    from collections import Sequence
    
    if isinstance(scale_or_shape, Real) and scale_or_shape > 0:
        scale = float(scale_or_shape)
        return (scale, scale), (ceil(scale*sh[0]), ceil(scale*sh[1]))

    if isinstance(scale_or_shape, Sequence) and len(scale_or_shape) == 2:
        if all((isinstance(ss, Integral) and ss > 0) or ss is None for ss in scale_or_shape) and any(ss is not None for ss in scale_or_shape):
            shape = list(scale_or_shape)
            if   shape[0] is None: shape[0], sz_dim = shape[1] * sh[0] / sh[1], 1
            elif shape[1] is None: shape[1], sz_dim = shape[0] * sh[1] / sh[0], 0
            else: sz_dim = None
            shape = (int(ceil(shape[0])), int(ceil(shape[1])))

            if sz_dim is not None:
                scale = shape[sz_dim] / sh[sz_dim]
                scale = (scale, scale)
            else:
                scale = (shape[0]/sh[0], shape[1]/sh[1])
            return scale, shape
        
        elif all(isinstance(ss, Real) and ss > 0 for ss in scale_or_shape):
            scale0, scale1 = float(scale_or_shape[0]), float(scale_or_shape[1])
            return (scale0, scale1), (ceil(scale0*sh[0]), ceil(scale1*sh[1]))
    
    raise ValueError("Invalid scale/output_shape")

def __im_and_out(im, out, out_shape):
    from numpy import require, empty

    # Check image
    if im.dtype.kind not in 'buif' or im.size == 0 or im.ndim < 2: raise ValueError("Invalid image")
    im = require(im, None, 'A')

    # Check or allocate output
    dt = im.dtype
    out_shape += im.shape[2:]
    if out is None:
        out = empty(out_shape, dt, order='F' if im.flags.fortran else 'C')
    elif out.shape != out_shape or out.dtype != dt:
        raise ValueError('Invalid output array')
    
    # Check logicals
    if dt.kind == 'b':
        from numpy import uint8
        im = im.view(uint8)*255 # unavoidable copy
        out = out.view(uint8)

    return im, out, dt

def __contributions(in_len, out_len, scale, kernel, kernel_width, antialiasing):
    from numpy import arange, floor, ceil, intp, where, logical_not, delete

    antialiasing = antialiasing and scale < 1
    
    # Use a modified kernel to simultaneously interpolate and antialias
    if antialiasing: kernel_width /= scale

    # Output-space coordinates.
    x = arange(out_len, dtype=float64)
    
    # Input-space coordinates. Calculate the inverse mapping such that 0.5 in output space maps to
    # 0.5 in input space, and 0.5+scale in output space maps to 1.5 in input space.
    x /= scale
    x += 0.5

    # What is the left-most pixel that can be involved in the computation?
    left = x - kernel_width/2
    left = floor(left, out=left).astype(intp)
    # left is the slice: int(-kernel_width/2) to int((out_len-1)/scale - kernel_width/2) stepping by 1/scale (kinda)
    
    # What is the maximum number of pixels that can be involved in the computation? Note: it's OK
    # to use an extra pixel here; if the corresponding weights are all zero, it will be eliminated
    # at the end of this function.
    P = int(ceil(kernel_width)) + 2

    # The indices of the input pixels involved in computing the k-th output pixel are in row k of the indices matrix.
    indices = left[:,None] + arange(P, dtype=intp)
    
    # The weights used to compute the k-th output pixel are in row k of the weights matrix.
    x = x[:,None] - indices
    if antialiasing:
        x *= scale
        weights = kernel(x)
        weights *= scale
    else:
        weights = kernel(x)
    
    # Normalize the weights matrix so that each row sums to 1.
    weights /= weights.sum(1, keepdims=True)
    
    # Clamp out-of-range indices; has the effect of replicating end-points.
    indices = indices.clip(0, in_len-1, out=indices)

    # If a column in weights is all zero, get rid of it.
    while not weights[:,0].any():
        if len(weights) == 1: return None, indices
        weights = weights[:,1:]
        indices = indices[:,1:]
    while not weights[:,-1].any():
        weights = weights[:,:-1]
        indices = indices[:,:-1]
    kill = where(logical_not(weights.any(0)))[0]
    if len(kill) > 0:
        weights = delete(weights, kill, axis=1)
        indices = delete(indices, kill, axis=1)

    # Detect if using nearest neighbor
    if (weights.ndim == 1 or weights.shape[1] == 1) and (weights == 1).all():
        return None, indices
    return weights, indices


########## Fast imresize ##########

def imresize_fast(im, out=None, nthreads=1):
    """
    Like imresize but with the following assumptions:
        scale is always (0.5, 0.5)
        method is always 'bicubic'
        antialiasing is always on
    But it does support everything else (2/3-D images, logical/integral/floating point types)
    """
    from numpy import empty, greater, ndindex
    from .__imresize import imresize_fast #pylint: disable=redefined-outer-name, no-name-in-module
    sh = im.shape
    im, out, dt = __im_and_out(im, out, ((sh[0]+1)//2, (sh[1]+1)//2))
    tmp = empty((out.shape[0], sh[1]), im.dtype, order='F' if im.flags.fortran else 'C')
    if im.ndim == 2:
        imresize_fast(im,    tmp,   nthreads)
        imresize_fast(tmp.T, out.T, nthreads)
    else:
        # NOTE: this just cycles through all the channels and does each one indendently
        # A faster method sometimes is to reshape the final dimension down and run it all at once
        # However, if this forces a copy, then it isn't so good afterall
        # If I find a good way to do this here, also should modify the non-fast version as well
        base = slice(None), slice(None)
        for idx in ndindex(sh[2:]):
            imresize_fast(im[base+idx], tmp,             nthreads)
            imresize_fast(tmp.T,        out[base+idx].T, nthreads)
    return greater(out, 128, out.view(dt)) if dt.kind == 'b' else out


########## Filters ##########
   
def cubic(x): # bicubic
    """
    See Keys, "Cubic Convolution Interpolation for Digital Image Processing," IEEE Transactions on
    Acoustics, Speech, and Signal Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
    """
    from numpy import less_equal, less, multiply, abs # pylint: disable=redefined-builtin
    absx = abs(x, out=x)
    absx2 = absx*absx
    absx3 = absx2*absx
    
    absx2 *= 2.5
    
    A = 1.5*absx3; A -= absx2; A += 1
    
    absx3 *= -0.5
    B = absx3; B += absx2; B -= multiply(4, absx, out=absx2); B += 2
    
    A *= less_equal(absx, 1, out=absx2)
    B *= less(1, absx, out=absx2)
    B *= less_equal(absx, 2, out=absx2)

    A += B
    return A

    #a = -0.5 # MATLAB's constant, OpenCV uses -0.75
    #return ((a+2)*absx3 - (a+3)*absx2 + 1) * (absx<=1) + \
    #       (    a*absx3 -   5*a*absx2 + 8*a*absx - 4*a) * logical_and(1<absx, absx<=2)
    ## Hardcoded a value
    #return (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + \
    #       (-.5*absx3 + 2.5*absx2 - 4*absx + 2) * logical_and(1<absx, absx<=2)
    
def box(x): # nearest
    from numpy import logical_and, less
    # logical_and(-0.5 <= x,x < 0.5)
    return logical_and(-0.5<=x, less(x,0.5, out=x), out=x)
def triangle(x): # bilinear
    from numpy import less_equal
    # (x+1)*logical_and(-1 <= x,x < 0) + (1-x)*logical_and(0 <= x,x <= 1)
    A = x + 1
    ls = x<0
    A *= ls
    A *= less_equal(-1,x,out=ls)
    B = 1 - x
    B *= less_equal(0,x,out=ls)
    B *= less_equal(x,1,out=ls)
    A += B
    return A
def lanczos2(x):
    """See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990, pp. 156-157."""
    from numpy import sin, pi, abs # pylint: disable=redefined-builtin
    abs(x, out=x)
    c = x<2
    x *= pi
    return (sin(x)*sin(0.5*x)+__eps)/((0.5*x*x)+__eps)*c
def lanczos3(x):
    """See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990, pp. 157-158."""
    from numpy import sin, pi, abs # pylint: disable=redefined-builtin
    abs(x, out=x)
    c = x<3
    x *= pi
    return (sin(x)*sin(1/3*x)+__eps)/((1/3*x*x)+__eps)*c
