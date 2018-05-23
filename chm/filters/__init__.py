"""
Filters used by the CHM segmentation algorithm.

These are based on the MATLAB functions originally used. They have been converted into an
object-oriented style. Every filter now can write directly to an output, use a region of an image
(so that padding isn't necessary), and is potentially multi-threaded.

The functions were standardized so now they all only take float64 2D image arrays (originally some
filters accepted color images or integer arrays, but not anymore). The output is now (features,H,W)
instead of (features,H*W), but this difference is just a reshape away.

Several filters were implemented very poorly originally, to the detriment of their output, so now
support a "compat" flag that, when given as True, tries to reproduce the original MATLAB outputs
more accurately, even if it means 'wrong' outputs or longer computations.

Some other changes, mostly very minor, are noted in CHANGED comments.

Jeffrey Bush, 2015-2016, NCMIR, UCSD
"""

from ._base import Filter, FilterBank
from .haar import Haar
from .hog import HOG
from .edge import Edge
from .frangi import Frangi
from .gabor import Gabor
from .sift import SIFT
from .intensity import Intensity

##### OPTIMIZATIONS #####
# Reduce padding:
#   SIFT (+3? has lots of correlate1d's)
#
# Could further optimize reflective padding by not actually adding padding but instead use the
# intrinsic padding abilities of the various scipy ndimage functions (no way to turn off their
# padding anyways...). This has been done for Intensity. However the other filters won't really
# benefit since currently Gabor cannot use it and it has by-far the largest padding requirement.
#
# These are in general optimized for feature-at-once (all pixels are done for a single feature)
# Some filters however operate pixel-at-once (all features are done for a single pixel)
# The following are pixel-at-once:
#   Haar (although only 2 features so not really a problem)
#   HOG  (36 features, a bit more of a problem)
#   SIFT (at least during the final normalization step and has 128 features!)
#
# Other optimizations/fixes:
#   Gabor don't precompute division?
#   SIFT  **lots of work needed including avoiding a divide-by-zero
#   Inten all DOUBLE_PTR_R and DOUBLE_PTR_CR should be aligned (A) but doing so causes illegal
#         argument core dumps...
#   Inten test when L_rad != R_rad to make sure it is right
