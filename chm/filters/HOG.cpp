// HOG - Histogram of oriented gradients features calculator
// Written by Jeffrey Bush, 2015-2016, NCMIR, UCSD
// Adapted from the code in HOG_orig.cpp with the following changes:
//  * Assumes that the image is always grayscale
//  * Params are hard-coded at compile time
//  * Uses doubles everywhere instead of float intermediates
//  * Does not require C++
//  * Much less memory is allocated
//  * Less looping in the second half
//  * Arrays are now used in C order instead of Fortran order
//  * Can divide initialization and running
//  * Supports running without implicit 0 padding
// So overall, faster, more accurate, and less memory intensive.

#define _USE_MATH_DEFINES
#include "HOG.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef NB_BINS
#define NB_BINS    9
#define CELLSZ_INV 0.125 // 1/cwidth (so cell_size == 8)
#define BLOCK_SIZE 2
#define ORIENT     M_PI  // originally a bool, if true we need 2*PI, if false we need just PI
#define CLIP_VAL   0.2
#endif

#ifdef H_INDEX
#undef H_INDEX
#endif
#define H_INDEX(y,x,b) (((y)*hist2 + (x))*NB_BINS + (b))

ssize_t HOG(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, const ssize_t n)
{
	ssize_t N;
	ssize_t HN = HOG_init(w, h, &N);
	if (n < N) { return -1; }
	dbl_ptr_ar H = (dbl_ptr_ar)malloc(HN*sizeof(double));
	if (H == NULL) { return -2; }
	HOG_run(pixels, w, h, out, H, 0);
	free(H);
	return N;
}

ssize_t HOG_init(const ssize_t w, const ssize_t h, ssize_t *n)
{
	const ssize_t hist1 = 2+(ssize_t)ceil(h*CELLSZ_INV - 0.5); // 4 for h = 15
	const ssize_t hist2 = 2+(ssize_t)ceil(w*CELLSZ_INV - 0.5); // 4 for w = 15
	*n = (hist1-BLOCK_SIZE-1)*(hist2-BLOCK_SIZE-1)*NB_BINS*BLOCK_SIZE*BLOCK_SIZE;
	return hist1*hist2*NB_BINS;
}

void HOG_run(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, dbl_ptr_ar H, const ssize_t padding)
{
	const ssize_t hist1 = 2+(ssize_t)ceil(h*CELLSZ_INV - 0.5);
	const ssize_t hist2 = 2+(ssize_t)ceil(w*CELLSZ_INV - 0.5);

	memset(H, 0, hist1*hist2*NB_BINS*sizeof(double));

	const ssize_t wp = w + 2*padding;
	pixels += padding + padding*wp; // move to the first non-padding pixel

	// Calculate gradients
	for (ssize_t y = 0; y < h; ++y)
	{
		const double cy = y*CELLSZ_INV + 0.5;
		const ssize_t y1 = (ssize_t)cy, y2 = y1 + 1;
		const double Yc = cy - y1 + 0.5*CELLSZ_INV;

		for (ssize_t x = 0; x < w; ++x)
		{
			double dx, dy;
			if (padding > 0)
			{
				dx = pixels[ +1] - pixels[-1]; // col after minus col before
				dy = pixels[-wp] - pixels[wp]; // row above minus row below
			}
			else
			{
				// out-of-bounds pixels are assumed to be 0
				dx = ((x!=w-1) ? pixels[+1] : 0) - ((x!=0)   ? pixels[-1] : 0);
				dy = ((y!=0)   ? pixels[-w] : 0) - ((y!=h-1) ? pixels[+w] : 0);
			}

			const double cx = x*CELLSZ_INV + 0.5;
            const ssize_t x1 = (ssize_t)cx, x2 = x1 + 1;
            const double Xc = cx - x1 + 0.5*CELLSZ_INV;

			const double grad_mag = sqrt(dx*dx + dy*dy);
			const double grad_or = (atan2(dy,dx) + ((dy<0)*ORIENT)) * (NB_BINS/ORIENT) + 0.5;
			ssize_t b2 = (ssize_t)grad_or, b1 = b2 - 1;
			const double Oc = grad_or - b2;
			if (b2 == NB_BINS) { b2 = 0; } else if (b1 < 0) { b1 = NB_BINS-1; }

			H[H_INDEX(y1,x1,b1)] += grad_mag*(1-Xc)*(1-Yc)*(1-Oc);
			H[H_INDEX(y1,x1,b2)] += grad_mag*(1-Xc)*(1-Yc)*(  Oc);
			H[H_INDEX(y2,x1,b1)] += grad_mag*(1-Xc)*(  Yc)*(1-Oc);
			H[H_INDEX(y2,x1,b2)] += grad_mag*(1-Xc)*(  Yc)*(  Oc);
			H[H_INDEX(y1,x2,b1)] += grad_mag*(  Xc)*(1-Yc)*(1-Oc);
			H[H_INDEX(y1,x2,b2)] += grad_mag*(  Xc)*(1-Yc)*(  Oc);
			H[H_INDEX(y2,x2,b1)] += grad_mag*(  Xc)*(  Yc)*(1-Oc);
			H[H_INDEX(y2,x2,b2)] += grad_mag*(  Xc)*(  Yc)*(  Oc);

			pixels += 1;
		}
		pixels += 2*padding; // skip padding
	}

	// Block normalization
	// This does 'L2-hys': L2-norm with clipping followed by re-normalization with L2-norm
	ssize_t out_i = 0;
	for (ssize_t x = 1; x < hist2-BLOCK_SIZE; ++x) // x = 1 only
	{
		for (ssize_t y = 1; y < hist1-BLOCK_SIZE; ++y) // y = 1 only
		{
			double block_norm = 0.0;
			for (ssize_t i = 0; i < BLOCK_SIZE; ++i) // i = 0 or 1
			{
				for (ssize_t j = 0; j < BLOCK_SIZE; ++j) // j = 0 or 1
				{
					for (ssize_t k = 0; k < NB_BINS; ++k) // k = 0, 1, 2, 3, 4, 5, 6, 7, 8
					{
						//
						const double val = H[H_INDEX(y+i,x+j,k)];
						block_norm += val * val;
					}
				}
			}

			const ssize_t out_start = out_i;
			double block_norm_2 = 0.0;
			if (block_norm > 0.0)
			{
				block_norm = 1.0 / sqrt(block_norm);
				for (ssize_t i = 0; i < BLOCK_SIZE; ++i)
				{
					for (ssize_t j = 0; j < BLOCK_SIZE; ++j)
					{
						for (ssize_t k = 0; k < NB_BINS; ++k)
						{
							double val = H[H_INDEX(y+i,x+j,k)] * block_norm;
							if (val > CLIP_VAL) { val = CLIP_VAL; }
							out[out_i++] = val;
							block_norm_2 += val * val;
						}
					}
				}
			}
			else { out_i += BLOCK_SIZE * BLOCK_SIZE * NB_BINS; }

			if (block_norm_2 > 0.0)
			{
				block_norm_2 = 1.0 / sqrt(block_norm_2);
				for (ssize_t i = out_start; i < out_i; ++i) { out[i] *= block_norm_2; }
			}
			else
			{
				memset(out+out_start, 0, (out_i-out_start) * sizeof(double));
			}
		}
	}
}
