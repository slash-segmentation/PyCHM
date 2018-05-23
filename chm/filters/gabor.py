"""
Gabor Filter. Implemented with FFTs.

NOTE: Also tested real-space ones and complex FFTs but both were slower.

This module can also be run using python -m chm.filters.gabor to create an FFTW wisdom cache.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from ._base import Filter

try:
    # See if we can use FFTW or give a warning that we can't
    import pyfftw #pylint: disable=unused-import
    _have_pyfftw = True
except ImportError:
    _have_pyfftw = False
    from warnings import warn
    warn('Unable to import pyfftw, using the NumPy FFT functions to calculate the Gabor filters. They are much slower.')

class Gabor(Filter):
    """
    Computes several different Gabor filters on the image producing using all combinations of the
    following parameters to create the kernel:
      sigma:      2, 3, 4, 5, 6
      lambdaFact: 2, 2.25, 2.5   (lambaFact = lamba / sigma)
      orient:     pi/6, pi/3, pi/2, 2pi/3, 5pi/6, pi, 7pi/6, 4pi/3, 3pi/2, 5pi/3, 11pi/6, 2pi
      gamma:      1
    Kernels using a phase of 0 and pi/2 are computed to represent the real and imaginary parts, and
    they combined using the L2 norm (complex magnitude) into a single feature, resulting in a total
    of 180 features.

    The original MATLAB code used real-space transforms while this one uses FFTs which makes it >10x
    faster (since the large FFT for the image itself can be calculated just once) and is just as
    accurate.

    Internally this can use either the NumPy FFT functions or the FFTW library if it is available.
    The FFTW version is faster and multi-threaded however it has a very large start-up cost to
    calculate the necessary 'wisdom' the first time it encouters a specific image size and number of
    threads combinations (resulting in an extra 5+ minutes of time required for the first call that
    is not present in any future calls). This information is saved between executions of the program
    as well in the file /tmp/pychm-pyfftw-wisdom (or similar) and can be copied to other machines
    with similar hardware.

    This uses O(4*im.size) intermediate memory. However if FFTW is used then O(3*im.size) of that is
    permanent across calls for each different image size and number of threads given.

    The original MATLAB code has a serious bug in it that it uses the imfilter function with a uint8
    input which causes the filter output to be clipped to 0-255 and rounded to an integer. This is a
    major problem as the Gabor filters have negative values (all of which are lost) and can produce
    results above the input range, along with losing lots of resolution in the data (from ~16
    significant digits to ~3). So, when compat is True, the data is put through the same
    simplification process. When it is False (the default), then much better results are produced.
    """
    def __init__(self, compat=False):
        super(Gabor, self).__init__(Gabor.__padding, len(Gabor.__filters))
        self.__compat = compat

    def __call__(self, im, out=None, region=None, nthreads=1):
        # INTERMEDIATE: im.shape (for either numpy or pyfttw)
        from itertools import izip
        from numpy import empty
        from ._base import get_image_region, round_u8_steps, hypot, copy

        padding, compat = Gabor.__padding, self.__compat
        im, region = get_image_region(im, padding, region, nthreads=nthreads)
        H, W = im.shape[0]-2*padding, im.shape[1]-2*padding # post-filtering shape
        if out is None: out = empty((len(Gabor.__filters), H, W), dtype=im.dtype)
        IF2 = empty((H, W), dtype=im.dtype)
        conv = Gabor.__get_convolve(im, nthreads)
        
        for IF1,(GF1,GF2) in izip(out, Gabor.__filters):
            start = len(GF1) // 2 + padding

            # Calculate the real and imaginary parts of the filter
            copy(IF1, conv(GF1)[start:H+start, start:W+start], nthreads)
            copy(IF2, conv(GF2)[start:H+start, start:W+start], nthreads)
            
            # Get the magnitude of the complex value
            if compat:
                round_u8_steps(IF1)
                round_u8_steps(IF2)
            hypot(IF1, IF2, IF1, nthreads)

        return out


    ##### 'Convolution' Methods #####
    # These actually use Fourier transforms. There are two versions of the function: one that uses
    # Numpy and one that uses FFTW (which must be installed separately and is used if possible). Both
    # functions take the master image and number of threads and return a funcion that will do the
    # actual convolution itself between the image and a kernel.
    @staticmethod
    def __get_convolve_numpy(im, _nthreads):
        # INTERMEDIATE:
        #     2*(im.shape[0],(im.shape[1]+1)//2) + (?,?)
        #     2*(im.shape[0],(im.shape[1]+1)//2) + (?,?)    [during call to conv]
        #     im.shape + (?,?)                              [during call to conv]
        from numpy.fft import rfft2, irfft2
        from ._base import next_regular
        im_fft_sh = tuple(next_regular(x) for x in im.shape)
        im = rfft2(im, s=im_fft_sh)
        def conv(K):
            from numpy import multiply
            F = rfft2(K, s=im_fft_sh)
            multiply(F, im, out=F)
            return irfft2(F, s=im_fft_sh)
        return conv

    @staticmethod
    def __get_convolve_pyfftw(im, nthreads):
        # INTERMEDIATES: for fftw: (these persist across calls, ~24 MB for 1000x1000)
        #           im.shape + (?,?)
        #        2*[(im.shape[0],im.shape[1]//2+1) + (?,?/2)] (complex)
        fft_im, fft_gf, ifft = Gabor.__get_fftw_plans(im.shape, nthreads)
        im = Gabor.__fftw(fft_im, im)
        ifft_out = ifft.output_array
        N = 1 / (ifft_out.shape[0] * ifft_out.shape[1])
        def conv(K):
            from numpy import multiply
            F = Gabor.__fftw(fft_gf, K)
            multiply(F, im, out=F)
            ifft.execute()
            multiply(ifft_out, N, out=ifft_out)
            return ifft_out
        return conv

    __get_convolve = (__get_convolve_pyfftw if _have_pyfftw else __get_convolve_numpy)


    ##### FFTW Helpers #####
    @staticmethod
    def __fftw(plan, src):
        """Copy the src into the plan's input array, padded with zeros, then run the plan."""
        from numpy import copyto

        # Copy data to upper-left corner, and zero-out bottom and right edges
        dst = plan.input_array
        dH, dW = dst.shape
        sH, sW = src.shape
        if dH < sH or dW < sW: raise ValueError()
        copyto(dst[:sH, :sW], src)
        dst[sH:,:].fill(0)
        dst[:sH,sW:].fill(0)

        plan.execute()
        return plan.output_array

    @staticmethod
    def __get_fftw_plans(sh, nthreads):
        """
        Create plans for running FFTW with Gabor filters. This caches the plans and buffers for
        same-sized images and number of threads.
        """
        from ._base import next_regular
        sh = tuple(next_regular(x) for x in sh)
        ID = sh + (nthreads,)
        plans = Gabor.__fftw_plans.get(ID)
        if plans is None:
            from numpy import float64, complex128
            from pyfftw import empty_aligned, FFTW
            fsh = sh[:-1] + (sh[-1]//2 + 1,)
            A = empty_aligned(sh,  float64)    # real-space image, kernel, or output
            B = empty_aligned(fsh, complex128) # fourier-space image
            C = empty_aligned(fsh, complex128) # fourier-space kernel
            fft_im = FFTW(A, B, (-2, -1), 'FFTW_FORWARD',  ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'), nthreads)
            fft_gf = FFTW(A, C, (-2, -1), 'FFTW_FORWARD',  ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'), nthreads)
            ifft   = FFTW(C, A, (-2, -1), 'FFTW_BACKWARD', ('FFTW_PATIENT', 'FFTW_DESTROY_INPUT'), nthreads)
            Gabor.__save_fftw_wisdom()
            Gabor.__fftw_plans[ID] = plans = (fft_im, fft_gf, ifft)
        return plans

    @staticmethod
    def __save_fftw_wisdom():
        """Exports the current FFTW wisdom to a temporary file for reuse if possible."""
        if not _have_pyfftw: return
        from pysegtools.general.io import umask
        from pyfftw import export_wisdom
        with umask(0), open(Gabor.__fftw_wisdom_file, 'wb') as f: f.write(export_wisdom()[0])
    
    @staticmethod
    def __load_fftw_wisdom():
        """Imports FFTW wisdom from a temporary file if possible."""
        from os.path import isfile
        if not _have_pyfftw or not isfile(Gabor.__fftw_wisdom_file): return False
        from pyfftw import import_wisdom
        with open(Gabor.__fftw_wisdom_file, 'rb') as f: wisdom = f.read()
        return import_wisdom((wisdom,'',''))[0]
  
    __fftw_plans = {}


    ##### Static Fields #####
    __filters = None
    __padding = None
    __fftw_wisdom_file = None

    @staticmethod
    def __get_filters():
        """
        Gets all of the convolution kernels to use for the Gabor fitler - a total of 360 filters.
        """
        # Not separated even when possible because FFT convolutions don't work that way
        # Result takes ~2MB, calculation takes ~30ms
        from numpy import pi, arange
        from itertools import product
        sigma = [2, 3, 4, 5, 6]
        lambdaFact = [2, 2.25, 2.5]
        orient = arange(pi/6, 2*pi+.1, pi/6) # [pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6, pi, 7*pi/6, 4*pi/3, 3*pi/2, 5*pi/3, 11*pi/6, 2*pi]
        phase = [0, pi/2] # doing 0 and pi/2 gets the real and imaginary parts of a filter at phase 0
        gamma = [1]
        return [[Gabor.__get_filter(s,o,l*s,p,g) for p in phase]
                for s, g, o, l in product(sigma, gamma, orient, lambdaFact)]

    @staticmethod
    def __get_filter(sigma,theta,lmbda,psi,gamma):
        """
        Returns a matrix with 2*ceil(3*sigma)+1 rows and columns with the values equal to:
            exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*cos(2*pi*x'/lmbda+psi)
        Where:
            x' =  x*cos(theta)+y*sin(theta)
            y' = -x*sin(theta)+y*cos(theta)
            x,y = distance from center of kernel
        With the constants:
            sigma - Gaussian envelope
            theta - orientation
            lmbda - wavelength
            psi   - phase
            gamma - aspect ratio
        """
        from numpy import ceil, ogrid, sin, cos, exp, pi, negative

        # Bounding box
        imax = int(ceil(max(1,3*sigma)))
        y,x = ogrid[-imax:imax+1,-imax:imax+1]

        # Rotation
        x_theta =  x*cos(theta)+y*sin(theta)
        y_theta = -x*sin(theta)+y*cos(theta)

        # Calculate exp(-((x_theta^2+y_theta^2*gamma^2)/(2*sigma^2)))*cos(2*pi/lmbda*x_theta+psi)
        ang = (2*pi/lmbda)*x_theta
        ang += psi
        cos(ang, out=ang)
        x_theta *= x_theta
        y_theta *= gamma
        y_theta *= y_theta
        x_theta += y_theta
        x_theta /= -2*sigma*sigma
        exp(x_theta, out=x_theta)
        x_theta *= ang
        if psi != 0:
            # TODO: this is a hack to make the Fourier transform versions work
            # (only effects compat mode)
            negative(x_theta, x_theta)
        x_theta.flags.writeable = False
        return x_theta

    @staticmethod
    def __static_init__():
        Gabor.__filters = Gabor.__get_filters()
        Gabor.__padding = max(len(GF[0]) // 2 for GF in Gabor.__filters)
        if _have_pyfftw:
            from os.path import join
            from tempfile import gettempdir
            from psutil import Process
            from os.path import basename
            hsh = hash(':'.join(mm.path for mm in Process().memory_maps() if 'libfftw' in basename(mm.path)))
            Gabor.__fftw_wisdom_file = join(gettempdir(), 'pychm-pyfftw-wisdom-%X'%hsh)
            Gabor.__load_fftw_wisdom()
            
    @staticmethod
    def __main__():
        import sys
        if not _have_pyfftw:
            print("Cannot create an FFTW wisdom cache since PyFFTW is not available.")
            sys.exit(0)
        if len(sys.argv) == 1:
            print(
"""This program creates an FFTW wisdom cache for use with PyCHM. You must provide
at least one comma-seperated list of height,width,num-threads or
height,width,levels,num-threads to create an FFTW wisdom cache from. Levels
specifies how many times to cut the image size in half automatically (default
is 0). You may specify any number of these and each will be used in creating
the cache. Additionally, the num-threads may not be just a number but a range
such as 1-24.

The cache will be saved to:
    %s
You can set the environmental variable TMPDIR, TEMP, or TMP to influence which
directory the cache will be saved to.

This cache is re-usable on all similar machines (hardware and software). The
filename contains a hash of the FFTW library in use which also needs to be the
same for re-using.""" % Gabor.__fftw_wisdom_file)
            sys.exit(0)
        for arg in sys.argv[1:]:
            try:
                vals = arg.split(',')
                if len(vals) == 4:
                    h,w,l,n = vals
                    l = int(l,10)
                else:
                    h,w,n = arg.split(',')
                    l = 0
                sh = int(h,10),int(w,10)
                n1,n2 = (int(n,10) for n in (n.split('-') if '-' in n else (n,n)))
                for i in xrange(l+1):
                    for n in xrange(n1, n2+1):
                        print('Calculating wisdom for %d,%d,%d...'%(sh[0],sh[1],n))
                        Gabor.__get_fftw_plans(sh, n)
                        Gabor.__fftw_plans.clear()
                    sh = (sh[0] + 1) // 2, (sh[1] + 1) // 2
            except ValueError:
                print('Invalid value "%s"'%arg)
        print('Completed')
                
Gabor.__static_init__()

if __name__ == "__main__": Gabor.__main__()
