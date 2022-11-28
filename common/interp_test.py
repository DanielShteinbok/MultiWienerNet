import one_shot_svd
import numpy as np

import load_PSFs
# TODO: import everything as per one_shot_svd and manage dependencies
#import 

def generate_unpadded_nointerp(psf_directory, metaman, img_dims, obj_dims, method="nearest"):
    """
    Generate the h and weights, where weights is NOT interpolated
    (for K PSFs, contains K entries)
    """
    # load the PSFs
    unpadded_psfs, indices = csv_psfs.load_from_dir_index(psf_directory)

    # flip and transpose PSFs to undo what Zemax does:
    #unpadded_psfs = np.transpose(np.flip(unpadded_psfs, (0,1)), axes=(1,0,2))
    unpadded_psfs = np.transpose(unpadded_psfs, axes=(1,0,2))

    # added this line to deal with nan-filled PSFs produced by Zemax
    unpadded_psfs[np.isnan(unpadded_psfs)] = 0

    # load the metaman:
    # for this function, we're just going to be passing the metaman directly in.
    # We want to have a high-level function that does things like instantiate it,
    # then call this function and others to orchestrate the "Mastermat" approach.
    #metaman = load_PSFs.MetaMan(psf_meta_path)

    # We're assuming that the pixel-dimensions of the object image will actually be the same as img_dims
    #find_pixel_on_obj = lambda x,y: (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))
    origins_pixel = {k: find_pixel_on_obj(*v, img_dims, obj_dims) for k, v in metaman.field_origins.items()}

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    #rank = 28
    rank = unpadded_psfs.shape[2] - 1
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    comps, weights_interp=svm.calc_svd_indexed_sized(unpadded_psfs, origins_pixel, indices, rank, img_dims, method=method)

    weights_norm = np.absolute(np.sum(weights_interp[weights_interp.shape[0]//2-1,weights_interp.shape[1]//2-1,:],0).max())
    weights = weights_interp/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=comps/np.linalg.norm(np.ravel(comps))

    return h, weights


#FIXME
#def interped_kernels(
