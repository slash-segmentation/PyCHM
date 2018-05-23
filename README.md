Python CHM Train and Test Algorithms
====================================

This package includes the CHM train and test algorithms written in Python with several
enhancements, including major speed and memory improvements, improved and added filters, and
improvements to the processing algorithm.

Models are not always compatible. MATLAB models created with the MATLAB `CHM_train` can be used
with the Python `chm.test` with only minor differences in output. However, Python models cannot be
used with the MATLAB `CHM_test`.


Installation
------------
The following libraries must be installed:
 * gcc and gfortran (or another C and Fortran compiler)
 * Python 2.7
 * Python headers

The following libraries are strongly recommended:
 * virtualenv
 * linear-algebra package including devel (in order of preference: MKL, ATLAS+LAPACK, OpenBLAS+LAPACK, any BLAS library)
 * devel packages for image formats you wish to read:
    * PNG: zlib (note: uncompressed always available)
    * TIFF: libtiff (note: uncompressed always available)
    * JPEG: libjpeg or libjpeg-turbo
    * etc...
 * hdf5 devel package for reading and writing modern MATLAB files
 * fftw devel package for faster FFT calculations

These can all be installed with various Python managers such as Anaconda, Enthought, Python(x,y),
or WinPython. On Linux machines they can be installed globally with `yum`, `apt-get`, or similar.
For example on CentOS-7 all of these can be installed with the following:

    yum install gcc gcc-gfortran python python-devel python-virtualenv \
                atlas atlas-devel lapack lapack-devel lapack64 lapack64-devel \
                zlib zlib-devel libtiff libtiff-devel libjpeg-turbo libjpeg-turbo-devel \
                hdf5 hdf5-devel fftw fftw-devel

The recommended way to install if from here is to create a Python virtual environment with all of
the dependent Python packages. On Linux machines, setting this up would look like:
    
    # Create the folder for the virtual environment
    # Adjust this as you see fit
    mkdir ~/virtenv
    cd ~/virtenv
    
    # Create and activate the virtual environment
    virtualenv .
    source bin/activate

    # Install some of the dependencies
    # Note: these can be skipped but greatly speeds up the other commands
    pip install numpy cython scipy
    
    # Install the devel pysegtools (and all dependencies)
    git clone git@github.com:slash-segmentation/segtools.git
    pip install -e segtools[PIL,MATLAB,OPT]

    # Install the devel PyCHM
    git clone git@github.com:slash-segmentation/CHM.git
    pip install -e CHM/python[OPT]

Since the pysegtools and CHM packages are installed in editable mode (`-e`), if there are updates
to the code, you can do the following to update them:

    cd ~/virtenv/segtools
    git pull

    cd ~/virtenv/CHM/python
    git pull
    ./setup.py build_ext --inplace # builds any changed Cython modules

For hints on getting this to work on a cluster (e.g. Comet) see the INSTALL guide for segtools.

	
CHM Test
--------
Basic usage is:

    python -m chm.test model input-image output-image <options>
    
General help will be given if no arguments (or invalid arguments) are provided

    python -m chm.test
    
The model must be a directory of a MATLAB model or a Python model file. For MATLAB models the
folder contains param.mat and MODEL\_level#\_stage#.mat. For Python models it is a single file.
*The MATLAB command line made this optional but now it is now mandatory.*

The CHM test program takes a single 2D input image, calculates the labels according to a model,
and saves the labels to the 2D output image. The images can be anything supported by the `imstack`
program for 2D images. The output-image can be given as a directory in which case the input image
filename and type are used. *The MATLAB command line allowed multiple files however now only a
single image is allowed now, to process more images you must write a loop in bash or similar.*

The CHM test program splits the input image into a bunch of tiles and operates on each tile
separately. The size of the tiles to process can be given with `-t #x#`, e.g. `-t 512x512`. The
default is 512x512. The tile size must be a multiple of 2^Nlevel of the model (typically Nlevel<=4,
so should be a multiple of 16). *The MATLAB command line called this option `-b`. Additionally, the
MATLAB command line program overlapped tiles (specified with `-o`) which is no longer needed or
supported. Finally, in MATLAB this was a critical quality vs speed option and now the default
should really always be used.*

Instead of computing the labels for every tile, tiles can be specified using either tile groups
(recommended) or individually. Tile groups use `-g` to specify which group to process and `-G` to
specify the number of groups. Thus when distributing the testing process among three machines the
processes would be run with `-g 1 -G 3`, `-g 2 -G 3` and `-g 3 -G 3` on the three machines. The
testing process will determine which tiles it is to process based on this information in a way that
reduces any extra work and making sure each process has roughly the same amount of work. The total
number of groups should not be larger than 30. Individual tiles can be specified using `-T #,#`,
e.g. `-T 0,2` computes the tile in the first column, third row. This option can be specified any
number of times to cause multiple tiles to be calculated at once. All tiles not calculated will
output as black (0). Any tile indices out of range are simply ignored. *The MATLAB command line
called this option `-t` and indexing started at 1 instead of 0 and did not support tile groups.*

By default the output image data is saved as single-byte grayscale data (0-255). The output data
type can be changed with `-d type` where `type` is one of `u8` (8-bit integer from 0-255), `u16`
(16-bit integer from 0-65535), `u32` (32-bit integer from 0-4294967295), `f32` (32-bit floating
point number from 0.0-1.0), or `f64` (64-bit floating-point number from 0.0-1.0). All of the other
data types increase the resolution of the output data. However the output image format must
support the data type (for example, PNG only supports u8 and u16 while TIFF supports of all the
types).

Finally, by default the CHM test program will attempt to use as much memory is available and all
logical processors available. If this is not desired, it can be tweaked using the `-n` and `-N`
options. The `-n` option specifies the number of tasks while `-N` specifies the number of threads
per task. Each additional task will take up significant memory (up to 2 GiB for default tiles and
Nlevel=4) while each additional thread per task doesn't effect memory significantly. However,
splitting the work into two tasks with one thread each will be much faster than having one task
with two threads. The default uses twice as many tasks as would fit in memory (since the maximum
memory usage is only used for a short period of time) up to the number of CPUs and then divides
the threads between all tasks. If only one of `-n` or `-N` is given, the other is derived based on
it. *The MATLAB command line only had the option to be multithreaded or not with `-s`.*

*The MATLAB command line option `-M` is completely gone as there is no need for MATLAB or MCR to be
installed. It also did histogram equalization by default. This is no longer supported at all and
should be done separately.*


CHM Train
---------
Basic usage is:

    python -m chm.train model inputs labels <options>

General help will be given if no arguments (or invalid arguments) are provided

    python -m chm.train

The CHM train program takes a set of input images and labels and creates a model for use with CHM
test. The model created from Python cannot be used with the MATLAB CHM test program. The input
images are specified as a single argument for anything that can be given to `imstack -L` (the value
may need to be enclosed in quotes to make it a single argument). They must be grayscale images. The
labels work similarily except that anything that is 0 is considered background while anything else
is considered a positive label. The inputs and labels are matched up in the order they are given
and paired images must be the same size.

The model is given as a path to a file for where to save the model to. If the option `-r` is also
specified and the path already contains (part of) a Python model, then the model is run in 'restart'
mode. In restart mode, the previous model is examined and as much of it is reused as possible. This
is useful for when a previous attempt failed partway through or when desiring to add additional
stages or levels to a model. If the filters are changed from the original model, any completed
stages/levels will not use the new filters but new stages/levels will. The input images and labels
must be the same when restarting.

The default number of stages and levels are 2 and 4 respectively. They can be set using `-S #` and
`-L #` respectively. The number of stages must be at least 2 while the number of levels must be at
least 1. Each additional stage will require very large amounts of time to compute, both while
training and testing. Additional levels don't add too much additional time to training or testing,
but do increase both. Typically, higher number of levels are required with larger structures and do
not contribute much for smaller structures. Some testing has shown that using more than 2 levels
does not contribute much to increased quality - at least for non-massive structures.

The filters used for generating features are, by default, the same used by MATLAB CHM train but
without extra compatibility. The filters can be adjusted using the `-f` option in various ways. The
available filters are `haar`, `hog`, `edge`, `gabor`, `sift`, `frangi`, and `intensity-<type>-#`
(where `<type>` is either `stencil` or `square` and `#` is the radius in pixels). To add filters to
the list of current filters do something like `-f +frangi`. To remove filters from the list of
current filters do something like `-f -hog`. Additionally, the list of filters can be specified
directly, for example `-f haar,hog,edge,gabor,sift,intensity-stencil-10` which would specify the
default set of filters. More then one `-f` option can be given and they will build off of each other.

Besides filters being used to generate the features for images, a filter is used on the 'contexts'
from the previous stages and levels to generate additional features. This filter can be specified
with `-c`, e.g. `-c intensity-stencil-7` specifies the default filter used. This only supports a
single filter, so `+`, `-`, or a list of filters cannot be given.

To speed up the training process the training data can be subsampled by using the `-s` option. If
the number of samples for a particular stage/level is over the value given, at most half of that
many positive and negative samples are kept. The default is to keep all samples. *MATLAB had a fixed
value of 6000000.*

The training algorithm also requires running the testing algorithm internally. If desired these
results can be saved using the option `-o`. This option takes anything that can be given to 
`imstack -S` although quotes may be required to make it a single argument. This means that if you
want to see the results of running CHM test on the training data you can get it for free. Like CHM
test, this supports the `-d` argument to specify the data type used to save the data.

If not all of the training data should be used, a set of mask images can be provided to select
which pixels will be considered during training. This is done with `-M masks` where `masks` can be
anything that can be given to `imstack -L` although quotes may be required to make it a single
argument. The number and size of the images must be the same as the input-images and label-images.
Any non-zero pixel in the mask will be used to train with. Note that all pixels, even if not
covered by the mask, are considered for generation of the features.

*The MATLAB command line option `-M` is completely gone as there is no need for MATLAB or MCR to be
installed. Additionally the option `-m` is now mandatory and listed first.*


Filters
-------

### Haar

Computes the Haar-like features of the image. This uses Viola and Jones' (2001) 2-rectangle
features (x and y directional features) of size 16. It uses their method of computing the integral
image first and then using fast lookups to get the features.

When used for MATLAB models the computations are bit slower but reduces the drift errors compared
to the MATLAB output. Technically it is slightly less accurate, but the numbers are in the range of
1e-8 off for a 1000x1000 tile.

### HOG

Computes the HOG (histogram of oriented gradients) features of the image.

The original MATLAB function used float32 values for many intermediate values so the outputs from
this filter are understandably off by up to 1e-7. The improved accuracy is used for MATLAB or 
Python models since it just adds more accuracy.

*TODO: more details - reference and parameters used and the new HOG*

### Edge

Computes the edge features of the image. This calculates the edge magnitudes by using convolution
with the second derivative of a Guassian with a sigma of 1.0 then returns all neighboring offsets
in a 7x7 block.

When used for MATLAB models, the tiles/images are padded with 0s instead of reflection. This is not
a good approach since a transition from 0s to the image data will result in a massive edge.

*TODO: more details - reference*

### Frangi

Computes the Frangi features of the image using the eigenvectors of the Hessian to compute the
likeliness of an image region to contain vessels or other image ridges, according to the method
described by Frangi (1998). This uses seven different Gaussian sigmas of 2, 3, 4, 5, 7, 9, and 11
each done with the image and the inverted image (looking for white and black ridges). The beta
value is fixed to 0.5 and c is dynamically calculated as half of the maximum Frobenius norm of all
Hessian matrices.

This is not used by default in Python models or at all for MATLAB models.

### Gabor

Computes several different Gabor filters on the image using all combinations of the following
parameters to create the kernels:
 * sigma:      2, 3, 4, 5, 6
 * lambdaFact: 2, 2.25, 2.5   (lambaFact = lamba / sigma)
 * orient:     pi/6, pi/3, pi/2, 2pi/3, 5pi/6, pi, 7pi/6, 4pi/3, 3pi/2, 5pi/3, 11pi/6, 2pi
 * gamma:      1
 * psi:        0 (phase)
The magnitude of the complex filtered image is used as the feature.

The original MATLAB code has a serious bug in it that it uses the `imfilter` function with an `uint8`
input which causes the filter output to be clipped to 0-255 and rounded to an integer before taking
the complex magnitude. This is a major problem as the Gabor filters have negative values (all of
which are set to 0) and can produce results above the input range, along with losing lots of
resolution in the data (from ~16 significant digits to ~3). So for MATLAB models the data is
simplified in this way, otherwise a much higher accuracy version is used.

*TODO: more details - reference*

### SIFT

*TODO*

### Intensity

Computes the neighborhood/intensity features of the image. Basically, moves the image around and
uses the shifted images values. Supports square or stencil neighborhoods of any radius >=1. In
compatibility mode these are always fixed to a particular type (stencil) and radius (10).
