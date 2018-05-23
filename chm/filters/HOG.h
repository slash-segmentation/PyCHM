// HOG - Histogram of oriented gradients features calculator
// Written by Jeffrey Bush, 2015-2016, NCMIR, UCSD

#ifdef _MSC_VER
#ifndef HAVE_SSIZE_T
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
#else
#include <sys/types.h>
#endif

#ifndef RESTRICT
#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__)
#define RESTRICT __restrict__
#elif (__STDC_VERSION__ >= 199901L)
#define RESTRICT restrict
#else
#define RESTRICT
#endif
#endif

#ifndef ALIGNED
#if defined(_MSC_VER)
#define ALIGNED(t, n) __declspec(align(n)) t
#elif defined(__GNUC__)
#define ALIGNED(t, n) t __attribute__((aligned(8)))
#else
#define ALIGNED(t, n) t
#endif
#endif

typedef ALIGNED(double, 8) dbl_algn;
typedef dbl_algn * RESTRICT dbl_ptr_ar; // aligned, restricted
typedef const dbl_algn * RESTRICT dbl_ptr_car; // const, aligned, restricted

#ifdef __cplusplus
extern "C" {
#endif

/**
 * HOG filtering.
 *   image is in pixels of w and h, must be C-contiguous
 *   out is where data is saved, it is n pixels long
 *   returns:
 *     -1 if out is not long enough
 *     -2 if temporary memory can't be allocated
 *     otherwise it returns the number of values written to out
 *
 * This essentially calls HOG_init then HOG_run.
 */
ssize_t HOG(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, const ssize_t n);

/**
 * Checks and initilization for HOG filtering.
 *   the image size is given in w/h
 *   the total room needed in the output is given in n (in number of pixels)
 *   returns the number of pixels needed in the temporary buffer
 */
ssize_t HOG_init(const ssize_t w, const ssize_t h, ssize_t *n);

/**
 * The core code for the HOG filtering.
 *   image is in pixels of (w+2*padding)*(h+2*padding), must be C-contiguous
 *     if padding is 0, data is implicitly padded with 0s; padding above 1 is pointless
 *   out is where data is saved, it must be at least n pixels long (the n from HOG_init)
 *   H is a temporary buffer with the number of elements returned by HOG_init
 */
void HOG_run(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, dbl_ptr_ar H, const ssize_t padding);

#ifdef __cplusplus
}
#endif
