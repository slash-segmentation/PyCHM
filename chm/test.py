#!/usr/bin/env python2
"""
CHM Image Testing - Runs CHM testing on an image. Can also be run as a command line program.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

__all__ = ["CHM_test", # operates over an entire image by breaking it into tiles
           "testCHM",  # operates on an entire image
           "CHM_test_max_mem"] # gives the expected amount of memory testCHM will use

def CHM_test(im, model, tilesize=None, tiles=None, ntasks=None, nthreads=None, ignore_bad_tiles=False):
    """
    CHM_test - CHM Image Testing
    Breaks an image into multiple tiles, runs CHM-test on each tile, and combines the results all
    back into a single image which it returns. This is optimized for parallelization of a single
    image by loading the image and various other items into shared memory and spawning subprocesses
    that each use the shared resources but work on their own tiles.
    
    im is a single image slice - the entire image as a numpy array, it may also be given as a
        pysegtools.image.ImageSource
    model is the path to the model or a Model object
    tilesize is the size of the tiles to process, either a:
            * a single integer for both width and height
            * a tuple of integers for width and height
        each value must be a multiple of 2^Nlevel (e.g. 16 when there are 4 levels) and the default
        value is 512x512
    tiles is a list of tile coordinates to process
        coordinates are in x,y format
        they go from 0,0 to ((im.shape[1]-1) // tilesize[0], (im.shape[0]-1) // tilesize[1])
        default is all tiles
        if not processing all tiles, the resulting image will be black wherever a tile was skipped
    ntasks is how many separate tasks to run in parallel, each processing a tile of the image
    nthreads is how many threads to use per task
        each additional parallel task takes up a lot of memory (up to 2 GB for 512x512 with
        Nlevel=4) while each additional CPU per task does not really increase memory usage, however
        running two tasks in parallel each with 1 CPU is faster than giving a single task two CPUs.
        The default is to run twice as many tasks as can fit in memory (since the max memory is
        only used for a short period of time) and divide the rest of the CPUs among the tasks
        if only one value is given, the other is calculated using its value
    ignore_bad_tiles is whether invalid tiles in the tiles argument cause an exception or are
        silently ignored, default is to throw an exception
    """
    from .utils import set_lib_threads
    set_lib_threads(1) # OpenMP must not use any threads before we fork the processes

    # Parse the arguments and get basic information
    im, model, tilesize, tiles, rgn, ntasks, nthreads = \
        __parse_args(im, model, tilesize, tiles, ntasks, nthreads, ignore_bad_tiles)
    offsets = __gen_offsets(rgn, model.nlevels)
    rgns, rgns_out = __gen_offset_regions(tiles, tilesize, offsets, im.shape, model.nlevels)
    
    # Allocate shared memory and run the processes
    mems = __alloc_shared_memory(im, rgn, model.nlevels)
    out = __run_chm_test_procs(mems, model, rgns, ntasks, nthreads)
    del mems
    
    # Make the output full-sized
    if out.shape == im.shape and __is_all_tiles(tiles, tilesize, im.shape): return out
    return __copy_regions(out, rgns_out, offsets[0], im.shape)
    
def __down_to_multiple(x, m):
    """Returns the multiple of `m` that is less than or equal to x."""
    return x - x % m
def __up_to_multiple(x, m):
    """Returns the multiple of `m` that is greater than or equal to x."""
    M = (m - x % m); return x + (M!=m)*M

def __parse_args(im, model, tilesize=None, tiles=None, ntasks=None, nthreads=None, ignore_bad_tiles=False):
    """
    Parse the arguments of the CHM_test function. See that function for the argument definitions.

    im is returned as-is or unwrapped from an ImageSource
    The model is loaded and returned as a Model object.
    tilesize is returned as a tuple that is in H,W order (even though it is given in W,H order).
    tiles is returned as a Nx2 array in y,x order & sorted (even though it is given in x,y order).
    region of the image needed is calculated and returned.
    ntasks and nthreads are adjusted as necessary and returned.
    """
    from pysegtools.images import ImageSource
    from .model import Model

    # Make sure the image is not an ImageSource
    if isinstance(im, ImageSource): im = im.data
    
    # Load the model
    model = Model.load(model)
    
    # Parse tile size
    tilesize = __get_tilesize(tilesize, model.nlevels)
    
    # Get the list of tiles to process
    tiles = __get_tiles(im.shape, tilesize, tiles, ignore_bad_tiles)
    
    # Get the image region
    rgn = __get_region(tiles, tilesize, model, im.shape)

    # Get ntasks and nthreads
    total_pixels = (rgn[2]-rgn[0])*(rgn[3]-rgn[1])
    ntasks, nthreads = __get_ntasks_and_nthreads(ntasks, nthreads, model, tilesize, len(tiles), total_pixels)

    return im, model, tilesize, tiles, rgn, ntasks, nthreads

def __get_tilesize(tilesize, nlvls=0):
    """
    Gets the tile size from the argument passed in. The argument is one of the following:
        None: result is 512x512
        Sequence of 2 values: they represent width and height of the tiles
        Scalar: same value for width and height of the tiles
    The width and height must be a multiple of 2^nlvls. The return value is height by width and
    is always an array.
    """
    from collections import Sequence
    from numpy import asarray
    if tilesize is None:
        tilesize = asarray([512, 512])
    elif isinstance(tilesize, Sequence) and len(tilesize) == 2:
        tilesize = asarray([int(tilesize[1]), int(tilesize[0])])
    else:
        tilesize = asarray([int(tilesize), int(tilesize)])
    if (tilesize % (1<<nlvls)).any(): raise ValueError('Tilesize must be a multiple of 2^%d'%nlvls)
    return tilesize

def __get_tiles(shape, tilesize, tiles=None, ignore_bad_tiles=False):
    """
    Get the tile coordinates from the CHM_test parameters. Always returns a Nx2 array of
    coordinates, all of which are valid for the image and tilesize and no duplicates. While the
    input tiles are specified as x,y, this function returns them in y,x. The tilesize argument is
    given in H,W order. Also, the tiles will be sorted by y then by x.
    """
    from numpy import array, indices, intp
    from itertools import izip
    max_y, max_x = ((x+ts-1)//ts for x,ts in izip(shape, tilesize))
    if tiles is None: return __sort_tiles(indices((max_y, max_x)).T.reshape((-1,2)))
    tiles = array(tiles, dtype=intp)
    if tiles.ndim != 2 or tiles.shape[1] != 2: raise ValueError('Invalid tile coordinates shape')
    good_tiles = (tiles >= 0).all(axis=1) & ((tiles[:,0] < max_x) & (tiles[:,1] < max_y))
    if not good_tiles.all():
        if ignore_bad_tiles: tiles = tiles[good_tiles]
        else: raise ValueError('Invalid tile coordinates')
    return __rm_dup_tiles(tiles[:,::-1])

def __sort_tiles(tiles):
    """Sort a list of tiles."""
    from numpy import lexsort
    return tiles.take(lexsort(tiles.T[::-1]), 0)

def __rm_dup_tiles(tiles):
    """Removes duplicate tiles from a list of tiles. Returns them sorted."""
    from numpy import concatenate
    tiles = __sort_tiles(tiles)
    return tiles.compress(concatenate(([True], (tiles[1:] != tiles[:-1]).any(axis=1))), 0)

def __get_region(tiles, tilesize, model, shape):
    """
    Get the region of the image required for all the listed tiles and their padding. The tiles are what is
    returned by __get_tiles and tilesize is what is returned by __get_tilesize. The shape is an array.
    """
    from numpy import array, minimum, maximum
    
    # Get the total amount of padding we need at level 0 so that we have enough for all levels
    pad = max(
        (max(model[s,l].image_filter.padding+model[s,l].context_filter.padding
             for s in xrange(1, model.nstages+(l==0)))+4+2)<<l
        for l in xrange(model.nlevels+1)
    )

    # Calculate the region
    mn = maximum(__down_to_multiple(tiles.min(0)   *tilesize-pad, tilesize), 0)
    mx = minimum(__up_to_multiple( (tiles.max(0)+1)*tilesize+pad, tilesize), shape)
    return array([mn[0], mn[1], mx[0], mx[1]])

def __get_ntasks_and_nthreads(ntasks, nthrds, model, tilesize, max_ntasks, total_pixels):
    """
    Gets the ntasks and nthreads arguments for CHM_test. Tries to maximize the number of tasks
    running based on the amount of memory, then maximize then number of threads based on the number
    of CPUs.
    """
    from pysegtools.general.utils import get_mem_usable, get_ncpus_usable
    base_mem = total_pixels*80/3 # expected amount of shared memory
    task_mem = CHM_test_max_mem(tilesize, model)//2
    ncpus = get_ncpus_usable()
    mem_avail = (get_mem_usable() - base_mem) // task_mem
    if ntasks is None and nthrds is None:
        ntasks = __clip(mem_avail, 1, min(max_ntasks, ncpus))
        nthrds = __clip(ncpus//ntasks, 1, ncpus)
    elif ntasks is None:
        nthrds = __clip(nthrds, 1, ncpus)
        ntasks = __clip(mem_avail, 1, min(max_ntasks, ncpus))
    elif nthrds is None:
        ntasks = __clip(int(ntasks),   1, max_ntasks)
        nthrds = __clip(ncpus//ntasks, 1, ncpus)
    else:
        ntasks = __clip(int(ntasks), 1, max_ntasks)
        nthrds = __clip(int(nthrds), 1, ncpus)
    return ntasks, nthrds
def __clip(x, a, b): return b if x > b else (a if x < a else x)
    
def __gen_shapes(sh, nlevels):
    """
    Generate a list of shapes that starts with the given shape (height by width) and then each
    subsequent shape is half the height and width (rounded up) from the one before it. The list has
    a total of Nlevel+1 items in it.
    """
    shapes = [None]*(nlevels+1)
    shapes[0] = sh
    for level in xrange(1, nlevels+1):
        shapes[level] = sh = (sh[0]+1)//2, (sh[1]+1)//2
    return shapes

def __is_all_tiles(tiles, tilesize, shape):
    """
    Checks if the tiles represent all tiles from the image. This make several sasumptions including
    that there are no duplicates (use __rm_dup_tiles if necessary first) and that there are no
    invalid tiles (which is guaranteed by __get_tiles).
    """
    from itertools import izip
    h,w = ((x+ts-1)//ts for x,ts in izip(shape, tilesize))
    return len(tiles) == h*w

def __gen_offsets(rgn, nlevels):
    """
    Generate a list of offsets that starts with the given region (y by x) and then each subsequent
    offset is halved from the one before it. The list has a total of Nlevel+1 items in it.
    """
    from numpy import hstack
    offsets = [hstack((rgn[:2], rgn[:2]))]
    for _ in xrange(1, nlevels+1): offsets.append((offsets[-1]+1)//2)
    return offsets

def __gen_padding_tiles(tiles, tilesize, shape):
    """
    Generate the tiles needed for padding the tiles that are being processed. The tiles should be as
    returned by __get_tiles, tilesize should be as returned by __get_tilesize, and the shape of the
    entire image and not just the working region. Returns a set of tiles that are just for padding.
    """
    from numpy import transpose, where, zeros
    from scipy.ndimage import binary_dilation, generate_binary_structure
    space = zeros((shape+tilesize-1)//tilesize, bool)
    space[tuple(tiles.T)] = True
    return transpose(where(binary_dilation(space, generate_binary_structure(2,2))^space))

def __gen_offset_regions(tiles, tilesize, offsets, shape, nlevels):
    """Generate all offset regions returning the regions to compute and the output regions."""
    from itertools import izip
    from numpy import vstack
    
    # Calculate the tiles that are needed for padding
    tiles_pad = __gen_padding_tiles(tiles, tilesize, shape)
                  
    # Calculate the regions for both the requested tiles and their padding
    regions = __gen_regions(tiles, tilesize, shape, nlevels)
    regions_pad = __gen_regions(tiles_pad, tilesize, shape, nlevels)
    
    # Offset the regions and combine regions and regions_pad
    rgns = [vstack((r,rp))-o for r,rp,o in izip(regions, regions_pad, offsets)]
    
    # Finished
    return rgns, (regions[0]-offsets[0])
    
def __gen_regions(tiles, tilesize, shape, nlevels):
    """Generate the regions for a series of a tiles for all levels."""
    from numpy import asarray, minimum, hstack
    shape,tilesize = asarray(shape), asarray(tilesize)
    regions    = [None]*(nlevels+1)
    regions[0] = hstack((tiles*tilesize, minimum((tiles+1)*tilesize, shape)))
    for level in xrange(1, nlevels+1):
        shape = (shape+1)//2
        if (tilesize%2).any(): tiles = __rm_dup_tiles(tiles // 2)
        else: tilesize = tilesize // 2
        regions[level] = hstack((tiles*tilesize, minimum((tiles+1)*tilesize, shape)))
    return regions

def __copy_regions(im, regions, offsets, full_shape):
    """Copies and `offsets` the `regions` from `im` to a new image of `full_shape`."""
    from .utils import copy
    from numpy import zeros
    out = zeros(full_shape)
    x = out[offsets[0]:,offsets[1]:]
    for rgn in regions: copy(x[rgn[0]:rgn[2],rgn[1]:rgn[3]], im[rgn[0]:rgn[2],rgn[1]:rgn[3]])
    return out

def __alloc_shared_memory(im, rgn, nlevels):
    """
    Allocate the shared memory needed and copying the image into im_sm. Returns the argmuments to
    give to __get_arrays.
    """
    from ctypes import c_double
    from multiprocessing import RawArray
    from numpy import float64, frombuffer
    from .utils import im2double

    shapes = __gen_shapes((rgn[2]-rgn[0],rgn[3]-rgn[1]), nlevels)
    sizes = [sh[0]*sh[1] for sh in shapes]

    # Move the image into shared memory (as floating-point)
    im_sm = RawArray(c_double, sizes[0])
    im2double(im, frombuffer(im_sm, float64).reshape(shapes[0]), rgn)

    # Allocate shared memory for the output/clabels, downsampled images, and contexts of all sizes
    # The ds_sm is used for several purposes
    # TODO: can these RawArrays be memmapped to a file?
    out_sms = [RawArray(c_double, sz) for sz in sizes]
    ds_sm   = RawArray(c_double, sizes[0])

    return im_sm, out_sms, ds_sm, shapes[0]

def __get_arrays(im_sm, out_sms, ds_sm, sh):
    """
    Get numpy arrays from shared memory for the image (at various downsamplings), outputs/clabels,
    and contexts (downsampled clables). The shape of the level 0 image is required as well. The
    output shared memory needs to be a list of shared memories. The others are a single shared
    memory. Returns a list of images at various downsamplings (one for each level), outputs for
    each level, and a list of contexts for each level (where "each level" means all Nlevel+1
    levels), and the temporary context when reaching level=0.
    """
    from numpy import frombuffer, float64
    shapes = __gen_shapes(sh, len(out_sms)-1)

    # im_sm is full-sized - it is only used for the original full-sized image
    # each out_sm is full-sized for its level - only ever used for the output of the various levels
    # ds_sm is full-sized - it is used for several purposes
    #                       for level > 0:
    #                           the first 'half' is used for the downsampled image (each full-sized)
    #                           the second 'half' is used for several contexts (each full-sized)
    #                       for stage != 1 and level == 0:
    #                           used as a copy of the output from the last level == 0
    #                       for stage == 1 and level == 0: not used
    # for a 25k by 25k image with 4 levels:
    #   im_sm requires 4.657 GiB
    #   ds_sm requires 4.657 GiB
    #   out_sms requires 6.203 GiB (4.657 + 1.164 + 0.291 + 0.073 + 0.018)
    #   TOTAL: 15.516 GiB
    #   A close estimation is 80/3*npixs (which is actually exact for an infinite number of levels)
    
    sizes = [sh[0]*sh[1] for sh in shapes]
    sh_sz = zip(shapes, sizes)
    
    ims = [frombuffer(im_sm, float64, sizes[0]).reshape(shapes[0])]+ \
          [frombuffer(ds_sm, float64, sz).reshape(sh) for sh,sz in sh_sz[1:]]
    outs = [frombuffer(o_sm, float64).reshape(sh) for o_sm,sh in zip(out_sms, shapes)]
    
    f64sz = outs[0].itemsize
    off = sizes[0] // 2
    cntxts = [[frombuffer(ds_sm, float64, sz, (off+i*sz)*f64sz).reshape(sh) for i in xrange(lvl)]
              for lvl,(sh,sz) in enumerate(sh_sz)]

    return ims, outs, cntxts, frombuffer(ds_sm, float64, sizes[0]).reshape(shapes[0])

def __run_chm_test_procs(mems, model, regions, ntasks, nthreads):
    """Starts ntasks processes running __run_chm_test_proc then calls __run_chm_test_parallel."""
    from multiprocessing import JoinableQueue, Process
    from time import sleep
    print("Running CHM test with %d task%s and %d thread%s per task" %
          (ntasks, 's' if ntasks > 1 else '', nthreads, 's' if nthreads > 1 else ''))
    nthreads_full = ntasks*nthreads

    # Start the child processes
    q = JoinableQueue()
    args = (mems, model, nthreads, q)
    processes = [Process(target=__run_chm_test_proc, name="CHM-test-%d"%p, args=args) for p in xrange(ntasks)]
    for p in processes: p.daemon = True; p.start()
    sleep(0)
    
    # Run the CHM-test in parallel
    try: out = __run_chm_test_parallel(mems, model, regions, q, processes, nthreads_full)
    except:
        __clear_queue(q)
        __kill_processes(processes)
        raise

    # Tell all processes we are done and make sure they all actually terminate
    for _ in xrange(ntasks): q.put_nowait(None)
    q.close()
    q.join()
    q.join_thread()
    for p in processes: p.join()

    # Done! Return the output image
    return out

def __run_chm_test_parallel(mems, model, regions, q, processes, nthreads_full):
    """
    Coordinates the parallel tasks usinq the JoinableQueue q. Also manages the downsampling
    of images and contexts in between levels utilizing nthreads_full.
    """
    from itertools import izip
    from .utils import MyDownSample, copy, set_lib_threads
    
    # At this point since all children processes have been spawned we can use OpenMP threads again
    set_lib_threads(nthreads_full)

    # View the shared memory as numpy arrays
    ims, outs, contexts, out_tmp = __get_arrays(*mems)

    # Go through each stage
    for m in model:
        stage, level = m.stage, m.level
        if level == 0:
            # Reset image and copy the level 0 output to temporary
            im = ims[0]
            copy(out_tmp, outs[0], nthreads_full)
        else:
            # Downsample image and calculate contexts
            im = MyDownSample(im, 1, ims[level], None, nthreads_full)
            for c,o in izip(contexts[level-1], contexts[level]): MyDownSample(c, 1, o, None, nthreads_full)
            MyDownSample(outs[level-1], 1, contexts[level][-1], None, nthreads_full)

        # Load the queue and wait
        for region in regions[level]: q.put_nowait((stage, level, tuple(region)))
        __wait_for_queue(q, stage, level, len(regions[level]), processes)

    # Done! Return the output image
    return outs[0]

def __wait_for_queue(q, stage, level, total_tiles, processes):
    """
    Waits for all items currently on the queue to be completed. If able, progress updates are
    outputed to stdout. This is dependent on two undocumented attributes of the JoinableQueue
    class. If they are not available, no updates are produced.

    If a processes crashes, this will raise an error after terminating the rest of the processes.
    """
    import time
    from sys import stdout
    msg = "Computing stage %d level %d..." % (stage, level)
    if hasattr(q, '_cond') and hasattr(q, '_unfinished_tasks'):
        # NOTE: this uses the undocumented attributes _cond and _unfinished_tasks of the
        # multiprocessing.Queue class. If they are not available, we just use q.join() but
        # then we can't check for failures or show progress updates.
        #pylint: disable=protected-access
        start_time = time.time() # NOTE: not using clock() since this thread spends most of its time waiting
        eol,timeout = ('\r',5) if stdout.isatty() else ('\n',60)
        stdout.write(msg+eol)
        stdout.flush()
        last_perc = 0
        with q._cond:
            while True:
                q._cond.wait(timeout=timeout)
                unfinished_tasks = q._unfinished_tasks.get_value()
                if unfinished_tasks == 0: print("%s Completed                          "%msg); break
                __check_processes(processes)
                perc = (total_tiles - unfinished_tasks) / total_tiles
                if perc > last_perc:
                    elapsed = time.time() - start_time
                    secs = elapsed/perc-elapsed
                    stdout.write("%s %4.1f%%   Est %d:%02d:%02d remaining     %s" %
                                 (msg, perc*100, secs//3600, (secs%3600)//60, round(secs%60), eol))
                    stdout.flush()
                    last_perc = perc
    else:
        stdout.write(msg)
        stdout.flush()
        q.join()
        print("Completed")
    __check_processes(processes)

def __check_processes(processes):
    """Raises an error if any process is not alive"""
    if any(not p.is_alive() for p in processes):
        raise RuntimeError('A process has crashed so we are giving up')

def __kill_processes(processes, timeout=0.1):
    """Interrupts, waits, then kills all processes that are still alive."""
    import signal
    from os import kill
    sig = getattr(signal, 'CTRL_C_EVENT', signal.SIGINT) # on Windows need to send signal.CTRL_C_EVENT
    for p in processes:
        if p.is_alive(): kill(p.pid, sig)
    for p in processes: p.join(timeout)
    for p in processes:
        if p.is_alive(): p.terminate()
        p.join()
        
def __clear_queue(q):
    """Clears a Queue object."""
    from Queue import Empty
    q.close()
    try:
        while True: q.get_nowait()
    except Empty: pass
        
def __run_chm_test_proc(mems, model, nthreads, q):
    """
    Runs a single CHM test sub-process. The first argument is a tuple to expand as the arguments to
    __get_arrays(), the model is a Model object, nthreads is the number of threads this process
    should be using, and q is the JoinableQueue of tiles to work on.

    This will call get() on the Queue until a None is retrieved, then it will stop. If the item is
    not None it must be a stage, level, and region to be processed.
    """
    # Get the images and outputs in shared memory as numpy arrays
    ims, outs, cntxts, out_tmp = __get_arrays(*mems)
    # Protect ourselves against accidental writes to these arrays
    # Can't do this since Cython doesn't support using read-only arrays
    #for im in ims: im.flags.writeable = False
    #for c in cntxts:
    #    for c in c: c.flags.writeable = False
            
    # Set the number of base library threads
    from .utils import set_lib_threads
    set_lib_threads(nthreads)

    # Process the queue
    prev_level = -1
    while True:
        try: tile = q.get()
        except KeyboardInterrupt: break
        try:
            if tile is None: break # All done!
            stage, level, region = tile
            if level != prev_level:
                im, out, mod = ims[level], outs[level], model[stage,level]
                if stage != 1 and level == 0:
                    get_contexts = __get_level0_contexts_func(im.shape, out_tmp, outs, mod, nthreads)
                else:
                    _contexts = cntxts[level]
                    get_contexts = lambda region:(_contexts,region)
            contexts, cntxt_rgn = get_contexts(region)
            X = mod.filter(im, contexts, None, region, cntxt_rgn, nthreads)
            del contexts
            out[region[0]:region[2],region[1]:region[3]] = mod.evaluate(X, nthreads)
            del X
        except KeyboardInterrupt: break 
        finally: q.task_done()

def __get_level0_contexts_func(shape, out_tmp, outs, model, nthreads):
    """Get a function for calculating the contexts and context region for stage!=1 level 0."""
    padding = model.context_filter.padding
    scales = [1<<lvl for lvl in xrange(1, len(outs))]
    IH, IW = shape
    def __get_level0_contexts(region):
        from .utils import MyUpSample
        
        # Get the region for the level 0 context and the shape of the output context
        T,L,B,R = region
        T,L,B,R = max(T-padding, 0), max(L-padding,0), min(B+padding,IH), min(R+padding,IW)
        cntxt_rgn = (region[0]-T, region[1]-L, region[2]-T, region[3]-L) # region of the output contexts
        H, W = B-T, R-L # overall shape of the output context
        
        # Get the contexts for each level
        regions = [(T//S, L//S, (B+S-1)//S, (R+S-1)//S) for S in scales]
        offs = [(T,L)] + [(T%S, L%S) for S in scales]
        contexts = [out_tmp] + [MyUpSample(c,lvl,None,rgn,nthreads) for lvl,(c,rgn) in enumerate(zip(outs[1:],regions),1)]
        contexts = [c[T:T+H,L:L+W] for c,(T,L) in zip(contexts,offs)]
        return contexts, cntxt_rgn
    return __get_level0_contexts

def testCHM(im, model, nthreads=1):
    """CHM testing across an entire image using the model without splitting it into tiles."""
    # CHANGED: the 'savingpath' (now called model) can now either be the path of the folder
    # containing the models or it can be an already loaded model
    # CHANGED: dropped Nstage, Nlevel, NFeatureContexts arguments - these are included in the model
    from .utils import MyDownSample, MyUpSample
    from .model import Model
    model = Model.load(model)
    nstages = model.nstages
    sh = im.shape
    clabels = None
    for sm in model:
        stage, level = sm.stage, sm.level
        if level == 0:
            imx = im
            contexts = [] if stage == 1 else \
                       [MyUpSample(c,i,nthreads=nthreads)[:sh[0],:sh[1]] for i,c in enumerate(clabels)]
            clabels = []
        else:
            imx = MyDownSample(imx, 1, nthreads=nthreads)
            contexts = [] if level == 1 else [MyDownSample(c, 1, nthreads=nthreads) for c in contexts]
            contexts.append(MyDownSample(clabels[-1], 1, nthreads=nthreads))
        X = sm.filter(imx, contexts, nthreads=nthreads)
        if stage == nstages and level == 0: del im, imx, contexts
        clabels.append(sm.evaluate(X, nthreads))
        del X
    return clabels[0]

def CHM_test_max_mem(tilesize, model):
    """
    Gets the theoretical maximum memory usage for CHM-test for a given tile size and a model, plus
    a small fudge factor that is larger than any additional random overhead that might be needed.

        tilesize a tuple of height and width for the size of a tile
        model    the model that will be used during testing

    This is calculated using the following formula:
        (476+57*(Nlevel+1))*8*tilesize + 200*8*tilesize + 20*8*tilesize
    Where 476 is the number of filter features, 57 is the number of context features generated at
    each level, 8 is the size of a double-precision floating point, 200 is number of discriminants
    used at level = 0 (when there are the most context features), and 20 is the number of
    discriminants per group at that level.

    Theoretical maximum memory usage is 1.92 GB (for 512x512 tiles and 4 levels). In practice I
    am seeing about +0.02 GB from this which is not too much overhead.

    Note: this actually asks the model for its evaluation memory per pixel, adds 128 bytes per
    pixel for overhead, then multiplies by the number of pixels.
    """
    from numpy import asarray
    tilesize = asarray(tilesize)
    sizes = [None]*(model.nlevels+1)
    sizes[0] = tilesize[0]*tilesize[1]
    for lvl in xrange(1, model.nlevels+1):
        if not (tilesize%2).any(): tilesize = tilesize // 2
        sizes[lvl] = tilesize[0]*tilesize[1]
    return max((m.classifier.evaluation_memory+128)*sizes[m.level] for m in model) # the +128 bytes/pixel is a fudge-factor

def get_tiles_for_group(grp, ngrps, imshape, tilesize=None):
    """
    Get the list of tiles that should be processed for a group given the number of groups, shape of
    the image (width-by-height), and the tile-size (width-by-height). For large numbers of groups
    (>30) this can take a significant amount of time. The group is given as a value from 1 to the
    number of groups. The returned tiles are in (x,y) format.
    """
    from itertools import product
    assert(1 <= grp <= ngrps and ngrps > 1)
    
    # Get the height and width of the image in tiles
    imshape = imshape[::-1]
    tilesize = __get_tilesize(tilesize)
    h,w = (imshape + tilesize - 1) // tilesize
    
    # Get the grouping (TODO: this is the step that takes a really long time)
    ns,hs = __best_grouping(h, w, ngrps)
    
    # Get the row and column of the group (starting at 0,0)
    row,col = 0,grp-1
    while ns[row] <= col: col -= ns[row]; row += 1
    ni,hi = ns[row],hs[row] # number of columns and height of the group's row

    # Get the width of the column in the row
    wi,r = divmod(w, ni) # wi is the min width in the row and r columns will have +1 widths
    ws = ([wi+1]*r+[wi]*(ni-r)) if r <= 1 else ([wi+1]*(r-1)+[wi]*(ni-r)+[wi+1])
    
    # Get the offset of the group
    yi,xi = sum(hs[:row]),sum(ws[:col])
    
    # Generate all of the tiles
    return list(product(xrange(xi,xi+wi),xrange(yi,yi+hi)))
    
def __best_grouping(H, W, G):
    """
    Finds the best way to arrange G groups in a W-by-H rectangle. The groups are all rectangles and
    arranged into rows with each row having a height and a number of columns. Returns the number of
    columns in each row and the height of each row.
    
    The algorithm used has a fairly poor assymtotic growth with respect to G but is independent of
    H and W. Values greater than 25 are likely to take several seconds.
    """
    from itertools import izip
    from heapq import nlargest

    m_min, g_min, h_min = None, None, None
    for g in __gen_ns(G):
        R = len(g)
        ends = (0,R-1)
        
        # Calculate the heights of each row
        h = [H*gi//G for gi in g] # base heights for each row
        rem = H - sum(h) # overall remainder that needs to be distributed across R, 0 <= rem < R
        # Add one to each of the rem heights with the largest remainders (preferring ends over middle)
        for _,_,i in nlargest(rem, ((H*gi%G, i in ends, i) for i,gi in enumerate(g))):
            h[i] += 1
        hx = [hi+(i!=0)+(i!=R-1) for i,hi in enumerate(h)] # add the borders to each of the heights
            
        # Calculate the penalty for this arrangement
        C_extra = 2*((W-2)*(R-1)-H+2*G-g[0]-g[-1]+sum(gi*hi for gi,hi in izip(g,h)))
        C_avg = (H*W+C_extra)/G
        err = [hxi*W//gi-C_avg for gi,hxi in izip(g,hx)] # the approximate error for each group
        rem = [W%gi for gi in g]
        m = C_extra + sum(
            abs(ei) if gi == 1 else
            max(2-ri,0)*abs(ei+hxi)+(gi-abs(ri-2))*abs(ei+2*hxi)+max(ri-2,0)*abs(ei+3*hxi)
            for gi,hxi,ri,ei in izip(g,hx,rem,err)
        )
        # Old method (drops some of the values constant across all arrangements)
        #m = (W-2)*R - g[0] - g[-1] + sum(
        #        gi*hi + (W%gi)*abs(hi*(W//gi)-area+hi) + (gi-W%gi)*abs(hi*(W//gi)-area) for hi,gi in izip(h,g)
        #)
        if m_min is None or m < m_min: m_min,g_min,h_min = m,g,h

    # Return the best number of columns per row and row heights
    return g_min,h_min

def __gen_ns(n):
    """
    Generates sequences of positive integers of the form:
        x_1, x_2, x_3, ..., x_m
    where:
        1 <= m <= n
        sum(x_i for i in 1..(m)) == n
        x_1 <= x_m
        x_i <= x_{i+1} for i in 2..(m-2)
    Runs in O(n^x) time where x seems to be some value between 10 and 11 which is pretty bad.
    However the original version of this (stars-and-bars algorithm) required O(2^n) which is
    significantly worse. Plus it required extra processing to filtering out all of the extra
    redundant sequences.
    """
    if   n == 1: yield (1,)
    elif n == 2: yield (1,1)
    else:
        # Trivial case for m == 1
        yield (n,)

        # Simple case: m == 2, x_1+x_m == n and x_1<=x_m
        for x in xrange(1, n//2+1): yield (x,n-x)

        for m in xrange(3, n):
            # x_1+x_m = 2 to n-m+2 with x_1<=x_m
            # the middle values must add up to the remainder and be in increasing order
            for sm in xrange(2, n-m+3):
                for ns in __gen_ns_mid(m-2, n-sm):
                    for x in xrange(1, sm//2+1):
                        yield (x,) + ns + (sm-x,)

        yield (1,)*n # trivial case for m == n

def __gen_ns_mid(m, n):
    """
    Generates sequences of positive integers of the form:
        x_1, x_2, x_3, ..., x_m
    where:
        sum(x_i for i in 1..m) == n
        x_i <= x_{i+1} for i in 1..m
    This is done recursively.
    """
    # Trivial and simple cases
    if m == 1: yield (n,)
    elif m == 2:
        for x in xrange(1, n//2+1): yield (x,n-x)
    else:
        # The first value can be from 1 to n // m and then recurse on the rest of the values
        for x in xrange(1, n//m+1):
            for ns in __gen_ns_mid(m-1, n-x):
                yield (x,) + ns

def __chm_test_main():
    """The CHM test command line program"""
    from numpy import iinfo, dtype
    from pysegtools.images.io import FileImageSource

    # Parse Arguments
    im_path, out_path, model, tilesize, group, tiles, ntasks, nthreads, dt = __chm_test_main_parse_args()

    # Get input image
    im = FileImageSource.open(im_path, True)
    
    # Convert group to tiles
    if group is not None:
        tiles = get_tiles_for_group(group[0], group[0], im.shape[::-1], tilesize)
    
    # Process input
    out = CHM_test(im, model, tilesize, tiles, ntasks, nthreads, True)
    
    # Save output
    if dtype(dt).kind == 'u':
        out *= iinfo(dt).max
        out.round(out=out)
    FileImageSource.create(out_path, out.astype(dt, copy=False), True).close()

def __chm_test_main_parse_args():
    """Parse the command line arguments for the CHM test command line program."""
    #pylint: disable=too-many-locals, too-many-branches, too-many-statements
    import os.path
    from sys import argv
    from getopt import getopt, GetoptError
    from pysegtools.general.utils import make_dir
    from pysegtools.images.io import FileImageSource
    
    from numpy import uint8, uint16, uint32, float32, float64
    dt_trans = {'u8':uint8, 'u16':uint16, 'u32':uint32, 'f32':float32, 'f64':float64}

    # Parse and minimally check arguments
    if len(argv) < 4: __chm_test_usage()
    if len(argv) > 4 and argv[4][0] != "-":
        __chm_test_usage("You provided more than 3 required arguments")

    # Check the model
    model = argv[1]
    if not os.path.exists(model): __chm_test_usage("Model cannot be found")

    # Check the input image
    im_path = argv[2]
    if not FileImageSource.openable(im_path, True):
        __chm_test_usage("Input image is of unknown type")

    # Check the output image
    out_path = argv[3]
    if out_path == '': out_path = '.'
    out_ended_with_slash = out_path[-1] == '/'
    out_path = os.path.abspath(out_path)
    if out_ended_with_slash:
        if not make_dir(out_path): __chm_test_usage("Could not create output directory")
    if os.path.isdir(out_path):
        if os.path.samefile(os.path.dirname(im_path), out_path):
            __chm_test_usage("If the output is to a directory it cannot be the directory the source image is in")
        out_path = os.path.join(out_path, os.path.basename(im_path))
    else:
        if len(os.path.splitext(out_path)[1]) < 2: out_path += os.path.splitext(im_path)[1]
        if not make_dir(os.path.dirname(out_path)): __chm_test_usage("Could not create output directory")
    if not FileImageSource.creatable(im_path, True):
        __chm_test_usage("Output image is of unknown type")

    # Get defaults for optional arguments
    tilesize = None
    group = None
    groups = None
    tiles = []
    dt = uint8
    ntasks = None
    nthreads = None

    # Parse the optional arguments
    try: opts, _ = getopt(argv[4:], "hg:G:t:T:d:n:N:")
    except GetoptError as err: __chm_test_usage(err)
    for o, a in opts:
        if o == "-h": __chm_test_usage()
        elif o == "-g":
            try: group = int(a, 10)
            except ValueError: __chm_test_usage("Group number must be a positive integer")
            if group <= 0: __chm_test_usage("Group number must be a positive integer")
        elif o == "-G":
            try: groups = int(a, 10)
            except ValueError: __chm_test_usage("Number of groups must be an integer >=2")
            if groups <= 1: __chm_test_usage("Number of groups must be an integer >=2")
        elif o == "-t":
            try:
                if 'x' in a: W,H = [int(x,10) for x in a.split('x', 1)]
                else:        W = H = int(a,10)
            except ValueError: __chm_test_usage("Tile size must be a positive integer or two positive integers seperated by an x")
            if W <= 0 or H <= 0: __chm_test_usage("Tile size must be a positive integer or two positive integers seperated by an x")
            tilesize = W,H
        elif o == "-T":
            try: C,R = [int(x,10) for x in a.split(',', 1)]
            except ValueError: __chm_test_usage("Tile position must be two positive integers seperated by a comma")
            if C < 0 or R < 0: __chm_test_usage("Tile position must be two positive integers seperated by a comma")
            tiles.append((C,R))
        elif o == "-d":
            a = a.lower()
            if a not in dt_trans: __chm_test_usage("Data type must be one of u8, u16, u32, f32, or f64")
            dt = dt_trans[a]
        elif o == "-n":
            try: ntasks = int(a, 10)
            except ValueError: __chm_test_usage("Number of tasks must be a positive integer")
            if ntasks <= 0: __chm_test_usage("Number of tasks must be a positive integer")
        elif o == "-N":
            try: nthreads = int(a, 10)
            except ValueError: __chm_test_usage("Number of threads must be a positive integer")
            if nthreads <= 0: __chm_test_usage("Number of threads must be a positive integer")
        else: __chm_test_usage("Invalid argument %s" % o)
    if group is not None:
        if groups is None: __chm_test_usage("Group number and number of groups must be used together")
        if len(tiles) != 0: __chm_test_usage("Groups cannot be used with tiles")
        if group > groups: __chm_test_usage("Group number cannot be greater than the number of groups")
        group = (group, groups)
    elif groups is not None: __chm_test_usage("Group number and number of groups must be used together")
    elif len(tiles) == 0: tiles = None
    
    return im_path, out_path, model, tilesize, group, tiles, ntasks, nthreads, dt

def __chm_test_usage(err=None):
    import sys
    if err is not None:
        print(err, file=sys.stderr)
        print()
    from . import __version__
    print("""CHM Image Testing Phase.  %s
    
%s model input output <optional arguments>
  model         The path to the model. For MATLAB models this is a folder that
                contains param.mat and MODEL_level#_stage#.mat. For Python
                models this a file that is alongside several and files like
                model_name-LDNN-#-#.npy.
  input         The input image to read.
                Accepts any 2D image accepted by `imstack` for reading.
  output        The output file or directory to save to.
                Accepts any 2D image accepted by `imstack` for writing.

Optional Arguments:
  -g group      The group of tiles to process, from 1 to the number of groups.
                given with -G.
  -G groups     The number of groups to divide the tiles into for distributed
                processing. Must be used with -g. All groups must use the same
                tile size and number of groups for this to work correctly.
                This will calculate how to divide up the groups optimally to
                reduce redundant work between groups and so that each group
                will perform the same amount of work. (NOTE: values larger than
                30 should probably not be used)
  -t tile_size  Set the tile size to use as WxH. By default the tile size is
                512x512. This must be a multiple of 2^Nlevel of the model.
  -T C,R        Specifies that only the given tiles be processed by CHM while
                all others simply output black. Each tile is given as C,R (e.g.
                2,1 would be the tile in the third column and second row). Can
                process multiple tiles by using multiple -T arguments. The tiles
                are defined by multiples of tile_size. A tile position out of
                range will be ignored. This cannot be used with -g/-G and in
                general that is a significantly better solution as it will
                optimize the workload of the groups.
  -d type       Set the output type of the data, one of u8 (default), u16, u32,
                f32, or f64; the output image type must support the data type.
  -n ntasks     How many separate tasks to run in parallel. Each task processes
                a single tile of the image at a time.
  -N nthreads   How many threads to use per task. Each additional parallel task
                takes up a lot of memory (up to 2 GB for 512x512 with Nlevel=4)
                while each additional thread per task does not increase memory
                usage, however running two tasks in parallel each with 1 thread
                is much faster than giving a single task two threads. Default is
                to run twice as many tasks as can fit in memory (since the max
                memory is only used for a short period of time) and divide the
                rest of the CPUs among the tasks. If only one value is given the
                other is calculated using it."""
          % (__version__, __loader__.fullname), file=sys.stderr) #pylint: disable=undefined-variable
    sys.exit(0 if err is None else 1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    __chm_test_main()
