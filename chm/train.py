#!/usr/bin/env python2
"""
CHM Image Training
Runs CHM training phase on a set of images. Can also be run as a command line program with
arguments.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

__all__ = ["CHM_train"]

def CHM_train(ims, lbls, model, subsamples=False, masks=None, nthreads=None, disp=True):
    """
    CHM_train - CHM Image Training
    
    Inputs:
        ims         ImageStack or a list of ImageSource objects to train on
        lbls        ImageStack or a list of ImageSource objects of the ground-truth data 
        model       Model object that has been created but not completely learned
        subsamples  Maximum number a samples to allow for a single stage/level, half of this
                    value is reserved for positive samples and half for negative samples
        masks       ImageStack or a list of ImageSource objects of the pixels to use from the
                    training data, by default uses all pixels
        nthreads    number of threads to use, defaulting to the number of physical CPUs/cores
        disp        if set to False will not display updates, defaults to True
    
    Returns the final set of labels calculated.
    """
    # Basic checks of images, labels, and masks and loads them for level 0
    ims0,lbls0,masks0 = __check_and_load(ims, lbls, masks)
        
    # Get parameters
    shapes = __get_all_shapes([im.shape for im in ims0], model.nlevels)
    nthreads = __get_nthreads(nthreads)
    disp = __print if disp else lambda s,l=0:None

    ########## CHM Train Core ##########
    contexts, clabels = None, None # contexts and clabels are indexed by image-index then level
    for sm in model:
        disp('Training stage %d level %d...'%(sm.stage,sm.level))

        ##### Update images, labels, contexts #####
        if sm.level == 0:
            ims, lbls = ims0[:], lbls0[:]
            if masks0 is not None: masks = masks0[:]
            contexts, clabels = __reset_contexts(clabels, shapes, nthreads)
        else:
            __downsample_images(ims, lbls, masks, contexts, clabels, nthreads)

        ##### Feature Extraction #####
        disp('Extracting features...', 1)
        X_full, Y, M = __extract_features(sm, ims, lbls, masks, contexts, nthreads)

        ##### Learning the classifier #####
        if sm.classifier.learned: disp('Skipping learning... (already complete)', 1)
        else:
            disp('Learning...', 1)
            # Apply masking and subsampling
            # OPT: parallelize compress or at least do something similar to LDNN's kmeans downsampling
            X,Y = (X_full,Y) if M is None else (X_full.compress(M, 1), Y.compress(M, 0))
            del M
            if subsamples is not False: X,Y = __subsample(X, Y, subsamples//2, nthreads=nthreads)
            # Note: always use 1 for nthreads during learning
            # OPT: allow multiple threads for clustering?
            sm.learn(X, Y, nthreads=nthreads) # TODO: the disp method should using the logging module
            del X, Y
            model.save()

        ##### Generate the outputs #####
        disp('Generating outputs...', 1)
        __generate_outputs(sm, X_full, shapes[sm.level], clabels, nthreads)
        del X_full
        disp('Accuracy: %f, F-value: %f, G-mean: %f'%__calc_performance(clabels, lbls), 1)
        
    ########## Cleanup and return final labels ##########
    disp('Complete!')
    del ims, ims0, lbls, lbls0, masks, masks0, contexts
    return [clbl[0] for clbl in clabels]

def __check_and_load(ims, lbls, masks=None):
    """
    Checks the images, labels, and masks for consistency. Raises a ValueError for any invalid
    inputs. Finally loads all of these ready for level 0.
    """
    from itertools import izip
    from .utils import im2double
    # Check
    if len(ims) < 1 or len(ims) != len(lbls): raise ValueError('You must provide at least 1 image set and equal numbers of training and label images')
    if any(len(im.dtype) > 1 or im.dtype.kind not in ('iufb') for im in ims): raise ValueError('Images must be grayscale')
    if any(len(lbl.dtype) > 1 or lbl.dtype.kind not in ('iufb') for lbl in lbls): raise ValueError('Labels must be grayscale')
    shapes = [im.shape for im in ims]
    if any(sh != lbl.shape for sh,lbl in izip(shapes,lbls)): raise ValueError('Labels must be the same shape as the corresponding images')
    if masks is not None:
        if len(ims) != len(masks): raise ValueError('The number of mask images must be equal to the number of training/label images')
        if any(len(mask.dtype) > 1 or mask.dtype.kind not in ('iufb') for mask in masks): raise ValueError('Masks must be grayscale')
        if any(sh != mask.shape for sh,mask in izip(shapes,masks)): raise ValueError('Masks must be the same shape as the corresponding images')
    # Load
    ims0   = [im2double(im.data) for im in ims]
    lbls0  = [lbl.data>0 for lbl in lbls]
    masks0 = None if masks is None else [mask.data>0 for mask in masks]
    return ims0, lbls0, masks0

def __get_nthreads(nthreads):
    """
    Gets the number of threads to use. If nthreads is None or 0 the result is min(ncpus, 4)
    otherwise the result is min(ncpus, nthreads) where ncpus is the physical number of CPUs.
    """
    from pysegtools.general.utils import get_ncpus_usable
    return min(get_ncpus_usable(), nthreads or 4)

def __print(s, depth=0):
    """
    Like print(...) but pre-pends the current timestamp, spaces dependent on the depth, and forces
    a flush.
    """
    import sys, datetime
    print('%s %s%s'%(str(datetime.datetime.utcnow())[:19], '  '*depth, s))
    sys.stdout.flush()
    
def __get_all_shapes(shapes, nlvls):
    """Get the downsampled image shapes from the image shapes at level 0"""
    all_shapes = [None]*(nlvls+1)
    all_shapes[0] = shapes
    for lvl in xrange(1, nlvls+1):
        shapes = [(((sh[0]+1)//2),((sh[1]+1)//2)) for sh in shapes]
        all_shapes[lvl] = shapes
    return all_shapes

def __reset_contexts(clabels, shapes, nthreads):
    """
    Get the reset context and clabels variabls when reaching level 0 (either the first time or any
    subsequent time). `clabels` is None if it is the first time, otherwise it is the previous
    clabels as a list of lists, indexed by image-index then level). The shapes are the shapes of
    all of the images as a list of lists, indexed by level then image-index. Returns the appropiate
    contexts and clabels values.
    """
    n = len(shapes[0]) # number of images
    cntxts = [[] for _ in xrange(n)] if clabels is None else \
             [[__upsample(c, lvl, shapes[0][i], nthreads) for lvl,c in enumerate(clbls)]
              for i,clbls in enumerate(clabels)]
    return cntxts, [[] for _ in xrange(n)]

def __upsample(im, L, sh, nthreads):
    """Like MyUpSample but constrains the final shape to sh"""
    from .utils import MyUpSample
    return MyUpSample(im,L,nthreads=nthreads)[:sh[0],:sh[1]]

def __downsample_images(ims, lbls, masks, contexts, clabels, nthreads):
    """
    Downsample the images, labels, and contexts when going from one level to the next. This
    operates on the lists in-place.
    """
    from .utils import MyDownSample, MyMaxPooling
    ims[:]  = [MyDownSample(im,  1, nthreads=nthreads) for im  in ims ]
    lbls[:] = [MyMaxPooling(lbl, 1, nthreads=nthreads) for lbl in lbls]
    if masks is not None:
        # TODO: instead of max-pooling for masks should down-sample be done then cutoff at 0.5?
        masks[:] = [MyMaxPooling(mask, 1, nthreads=nthreads) for mask in masks]
    if len(clabels[0]) == 0: contexts[:] = [[] for _ in ims]
    for i,clbl in enumerate(clabels): contexts[i].append(clbl[-1])
    contexts[:] = [[MyDownSample(c, 1, nthreads=nthreads) for c in cntxts] for cntxts in contexts]

def __extract_features(submodel, ims, lbls, masks, contexts, nthreads):
    """
    Extract all features from the images into a feature vector. Returns X (feature vector), Y
    (labels), and M (mask or None if no masks).
    """
    #pylint: disable=too-many-locals
    from itertools import izip
    from numpy import empty
    from .utils import copy_flat
    ends = list(__cumsum(im.shape[0]*im.shape[1] for im in ims))
    npixs, nfeats = ends[-1], submodel.features
    start = 0
    X = empty((nfeats, npixs))
    Y = empty(npixs, bool)
    for im,lbl,cntxts,end in izip(ims, lbls, contexts, ends):
        submodel.filter(im, cntxts, X[:,start:end].reshape((nfeats,)+im.shape), nthreads=nthreads)
        copy_flat(Y[start:end], lbl, nthreads)
        start = end
    if masks is None: return X, Y, None
    start = 0
    M = empty(npixs, bool)
    for mask,end in izip(masks, ends):
        copy_flat(M[start:end], mask, nthreads)
        start = end
    return X, Y, M

def __cumsum(itr):
    """Like `numpy.cumsum` but takes any iterator and results in an iterator."""
    total = 0
    for x in itr: total += x; yield total
    
def __subsample(X, Y, n=3000000, nthreads=1):
    """
    Sub-sample the data. If the number of pixels is greater than 2*n then at most n rows are kept
    where Y is True and n rows from where Y is False. The rows kept are selected randomly.

    X is the feature vectors, a matrix that is features by pixels.
    Y is the label data, and has a True or False for each pixel.

    Returns the possibly subsampled X and Y.
    """
    # OPT: use nthreads and improve speed (possibly similar to LDNN's kmeans downsampling)
    # Currently takes ~25 secs on a 12,500,000 element dataset that is reduced to 3,116,282
    npixs = len(Y)
    if npixs <= 2*n: return X, Y

    from numpy import zeros, flatnonzero
    from .__shuffle import shuffle_partial #pylint: disable=no-name-in-module
    
    n_trues = Y.sum()
    n_falses = npixs-n_trues
    keep = zeros(npixs, bool)
    if n_trues > n:
        ind = flatnonzero(Y)
        shuffle_partial(ind, n)
        keep[ind[-n:]] = True
        del ind
    else: keep |= Y
    if n_falses > n:
        ind = flatnonzero(~Y)
        shuffle_partial(ind, n)
        keep[ind[-n:]] = True
        del ind
    else: keep |= ~Y
    return X[:,keep], Y[keep]

def __generate_outputs(submodel, X, shapes, clabels, nthreads):
    """
    Generates the outputs of the feature vector X from images that have the given shapes. The
    results are stored in the clabels list. The clabels list is a list-of-lists with the first
    index being the image index and the second being the level.
    """
    from itertools import izip
    start = 0
    for sh,clbl in izip(shapes, clabels):
        end = start + sh[0]*sh[1]
        clbl.append(submodel.evaluate(X[:,start:end], nthreads).reshape(sh))
        start = end

def __calc_performance(clabels, lbls):
    """
    Calculates and returns the performance (pixel accuracy, F-value, and G-mean) of the clabels
    (predicted) vs labels (ground-truth) image data.
    """
    from chm.utils import calc_confusion_matrix, calc_accuracy, calc_fvalue, calc_gmean
    from pysegtools.images import ImageStack
    from pysegtools.images.filters.threshold import ThresholdImageStack
    predicted = ThresholdImageStack(ImageStack.as_image_stack([clbl[-1] for clbl in clabels]), 'auto-stack')
    confusion_matrix = calc_confusion_matrix(predicted, lbls)
    return calc_accuracy(*confusion_matrix), calc_fvalue(*confusion_matrix), calc_gmean(*confusion_matrix)


def __chm_train_main():
    """The CHM train command line program"""
    # Parse Arguments
    ims, lbls, model, subsamples, masks, output, dt, nthreads = __chm_train_main_parse_args()

    # Process input
    out = CHM_train(ims, lbls, model, subsamples, masks, nthreads)

    # Save output
    if output is not None:
        from pysegtools.images.io import FileImageStack
        FileImageStack.create_cmd(output, [__adjust_output(o,dt) for o in out], True).close()

def __adjust_output(out, dt):
    """Adjusts the output array to match the given data type, scalling unsigned integral types."""
    from numpy import iinfo, dtype
    if dtype(dt).kind == 'u':
        out *= iinfo(dt).max
        out.round(out=out)
    return out.astype(dt, copy=False)

def __chm_train_main_parse_args():
    """Parse the command line arguments for the CHM train command line program."""
    #pylint: disable=too-many-locals, too-many-branches, too-many-statements
    from sys import argv
    from collections import OrderedDict
    from getopt import getopt, GetoptError
    from pysegtools.images.io import FileImageStack
    from .filters import FilterBank, Haar, HOG, Edge, Gabor, SIFT, Intensity
    from .model import Model
    from .ldnn import LDNN

    from numpy import uint8, uint16, uint32, float32, float64
    dt_trans = {'u8':uint8, 'u16':uint16, 'u32':uint32, 'f32':float32, 'f64':float64}

    # Parse and minimally check arguments
    if len(argv) < 4: __chm_train_usage()
    if len(argv) > 4 and argv[4][0] != "-":
        __chm_train_usage("You provided more than 3 required arguments")
        
    # Get the model path
    path = argv[1]

    # Open the input and label images
    ims = FileImageStack.open_cmd(argv[2])
    lbls = FileImageStack.open_cmd(argv[3])

    # Get defaults for optional arguments
    nstages, nlevels = 2, 4
    # TODO: SIFT should not need compat mode here!
    fltrs = OrderedDict((('haar',Haar()), ('hog',HOG()), ('edge',Edge()), ('gabor-compat',Gabor(True)),
                         ('sift',SIFT(True)), ('intensity-stencil-10',Intensity.Stencil(10))))
    cntxt_fltr = Intensity.Stencil(7)
    norm_method = 'median-mad'
    masks = None
    subsamples = False
    output, dt = None, uint8
    restart = False
    nthreads = None

    # Parse the optional arguments
    try: opts, _ = getopt(argv[4:], "rS:L:f:c:s:M:o:d:n:N:")
    except GetoptError as err: __chm_train_usage(err)
    for o, a in opts:
        if o == "-S":
            try: nstages = int(a, 10)
            except ValueError: __chm_train_usage("Number of stages must be an integer >= 2")
            if nstages < 2: __chm_train_usage("Number of stages must be an integer >= 2")
        elif o == "-L":
            try: nlevels = int(a, 10)
            except ValueError: __chm_train_usage("Number of levels must be a positive integer")
            if nlevels <= 0: __chm_train_usage("Number of levels must be a positive integer")
        elif o == "-f":
            if len(a) == 0: __chm_train_usage("Must list at least one filter")
            if a[0] == '+':
                fltrs.update((f,__get_filter(f)) for f in a[1:].lower().split(','))
            elif a[0] == '-':
                for f in a[1:].lower().split(','): fltrs.pop(f)
            else:
                fltrs = OrderedDict((f,__get_filter(f)) for f in a.lower().split(','))
        elif o == "-c":
            cntxt_fltr = __get_filter(a.lower())
        elif o == "-n":
            a = a.lower()
            if a not in ('none', 'min-max', 'mean-std', 'median-mad', 'iqr'): # TODO: use model.__norm_methods
                __chm_train_usage("The norm method must be one of none, min-max, mean-std, median-mad, or iqr")
            norm_method = a
        elif o == "-s":
            try: subsamples = int(a, 10)
            except ValueError: __chm_train_usage("Number of subsamples must be an integer >= 1000000")
            if subsamples < 1000000: __chm_train_usage("Number of subsamples must be an integer >= 1000000")
        elif o == "-M":
            masks = FileImageStack.open_cmd(a)
        elif o == "-o":
            output = a
            FileImageStack.create_cmd(output, None, True)
        elif o == "-d":
            a = a.lower()
            if a not in dt_trans: __chm_train_usage("Data type must be one of u8, u16, u32, f32, or f64")
            dt = dt_trans[a]
        elif o == "-r": restart = True
        elif o == "-N":
            try: nthreads = int(a, 10)
            except ValueError: __chm_train_usage("Number of threads must be a positive integer")
            if nthreads <= 0: __chm_train_usage("Number of threads must be a positive integer")
        else: __chm_train_usage("Invalid argument %s" % o)
    if len(fltrs) == 0: __chm_train_usage("Must list at least one filter")
    fltrs = FilterBank(fltrs.values())
    classifier = Model.nested_list(nstages, nlevels, lambda s,l:LDNN(LDNN.get_default_params(s,l)))
    model = Model.create(path, nstages, nlevels, classifier, fltrs, cntxt_fltr, norm_method, restart)
    return (ims, lbls, model, subsamples, masks, output, dt, nthreads)

def __get_filter(f):
    """
    Gets a Filter from a string argument. The string must be one of haar, hog, edge, frangi, gabor,
    sift, or intensity-[square|stencil]-#.
    """
    from .filters import Haar, HOG, Edge, Frangi, Gabor, SIFT, Intensity
    if f.startswith('intensity-'):
        f = f[10:]
        if f.startswith('square-'):    f = f[7:]; F = Intensity.Square
        elif f.startswith('stencil-'): f = f[8:]; F = Intensity.Stencil
        try: f = int(f, 10)
        except ValueError: __chm_train_usage("Size of intensity filter must be a positive integer")
        if f <= 0: __chm_train_usage("Size of intensity filter must be a positive integer")
        return F(f)
    filters = {'haar':Haar, 'hog':HOG, 'edge':Edge, 'frangi':Frangi, 'gabor':Gabor, 'sift': SIFT}
    compat = f.endswith('-compat')
    if compat: f = f[:-7]
    if f not in filters: __chm_train_usage("Unknown filter '%s'"%f)
    if compat:
        try: return filters[f](compat=True)
        except TypeError: __chm_train_usage("Unknown filter '%s-compat'"%f)
    return filters[f]()

def __chm_train_usage(err=None):
    import sys
    if err is not None:
        print(err, file=sys.stderr)
        print()
    from . import __version__
    print("""CHM Image Training Phase.  %s

%s model inputs labels <optional arguments>
  model         The file where the model data is saved to.
  inputs        The input image(s) to read
                Accepts anything that can be given to `imstack -L` except that
                the value must be quoted so it is a single argument.
  labels        The label/ground truth image(s) (0=background)
                Accepts anything that can be given to `imstack -L` except that
                the value must be quoted so it is a single argument.
  The inputs and labels are matched up in the order they are given and paired
  images must be the same size.

Optional Arguments:
  -S nstages    The number of stages of training to perform. Must be >=2.
                Default is 2.
  -L nlevels    The number of levels of training to perform. Must be >=1.
                Default is 4.
  -f filters... Comma separated list of filters to use on each image to generate
                the features. Options are haar, hog, edge, gabor, sift, frangi,
                and intensity-<type>-#. For the intensity filter, the <type> can
                be either stencil or square and the # is the size in pixels. If
                the whole argument starts with + then it will add them to the
                defaults. If the argument starts with - then they will be
                removed from the defaults.
                Default: haar,hog,edge,gabor-compat,sift,intensity-stencil-10
  -c filter     The filter used to generate features for context images. This
                takes a single filter listed above.
                Default is intensity-stencil-7
  -n norm       Normalize each feature in the data using the given method.
                The methods that are available are:
                    none: no normalization is performed
                    min-max: minimum goes to 0, maximum goes to 1
                    mean-std: mean to 0.5 and mean-/+2.5*std to 0 and 1
                    median-mad: median to 0.5 and median-/+2.5*MAD to 0 and 1
                        where MAD is the standardized median abs diff (default)
                    iqr: Q1-1.5*IQR to 0 and Q3+1.5*IQR to 1
  -M mask       Specify a mask of the input images and labels for which pixels
                should be used during training. The default is to use all
                pixels. The mask can be anything that can be given to
                `imstack -L` except that the value must be quoted so it is a
                single argument. It must contain the same number of images as
                input and label along with each image being the same size. Any
                non-zero pixel in the mask represent a point used for training,
                either a positive or negative label.
  -s subsamples Subsample the data. If the number of samples for a particular
                stage/level is over the value given, at most half of that many
                positive and negative samples are kept. The default is keeping
                all samples.
  -o output     Output the results of testing the model on the input data to the
                given image. Accepts anything that is accepted by `imstack -S`
                in quotes. The data is always calculated anyways so this just
                causes it to be saved in a usable format.
  -d type       Set the output type of the data, one of u8 (default), u16, u32,
                f32, or f64; the output image type must support the data type.
  -r            Restart an aborted training attempt. This will restart just
                after the last completed stage/level. You must give the same
                inputs and labels as before for the model to make sense. However
                you do not need to give the same filters, stages, or levels. The
                model will be adjusted as necessary.
  -N nthreads   How many threads to use for feature extraction and output
                generation. Default is at most 4 threads or the number of 
                physical CPUs. Learning is never multithreaded and multiple
                tasks are not supported at all."""
          % (__version__, __loader__.fullname), file=sys.stderr) #pylint: disable=undefined-variable
    sys.exit(0 if err is None else 1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    __chm_train_main()
