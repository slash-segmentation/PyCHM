"""
Cython Shuffle function definition. See the accompying PYX file for more information.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from npy_helper cimport intp

cpdef void shuffle(intp[::1] arr) nogil
cpdef void shuffle_partial(intp[::1] arr, intp sub) nogil
