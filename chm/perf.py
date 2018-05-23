#!/usr/bin/env python2
"""
Calculate the accuracy/performance of a set of images compared to the ground truth data.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

def main():
    from .utils import calc_confusion_matrix, calc_accuracy, calc_fvalue, calc_gmean
    from pysegtools.images.io import FileImageStack
    from pysegtools.images.filters.threshold import ThresholdImageStack

    import sys
    if len(sys.argv) != 3:
        print("Requires 2 arguments: a stack of predicted images and a stack of ground-truth images")
        sys.exit(1)
    
    # Open the image stacks
    predicted = FileImageStack.open_cmd(sys.argv[1])
    ground_truth = FileImageStack.open_cmd(sys.argv[2])
    
    # Wrap them in a thresholding image stack if they are not already boolean
    if not predicted.is_dtype_homogeneous or predicted.dtype != bool:
        predicted = ThresholdImageStack(predicted, 'auto-stack')
    if not ground_truth.is_dtype_homogeneous or ground_truth.dtype != bool:
        ground_truth = ThresholdImageStack(ground_truth, 1)

    confusion_matrix = calc_confusion_matrix(predicted, ground_truth)
    print("TP = %d, TN = %d, FP = %d, FN = %d"%confusion_matrix)
    print("Accuracy = %f"%calc_accuracy(*confusion_matrix))
    print("F-value  = %f"%calc_fvalue(*confusion_matrix))
    print("G-mean   = %f"%calc_gmean(*confusion_matrix))

if __name__ == "__main__": main()
