import csv_psfs
import svd_model as svm
import numpy as np
import scipy

# from tensorflow directory
import sys
sys.path.append("../tensorflow")
import load_PSFs

def generate_model(psf_directory, psf_meta_path, img_dims, obj_dims):
    """
    Generate the eigen-PSFs and weights for the entire field of view.

    Parameters:
        psf_directory: str, path to directory with the PSF CSV files therein
        psf_meta_path: str, path to meta csv file which describes the locations and shifts of the PSFs
        img_dims: (int height, int width), the dimensions of the sensor, in pixels
        obj_dims: (int height, int width), the dimensions of the field of view, in microns
    Returns:
        h: the eigen-PSFs
        weights: the weights at each pixel of the image
    """
    # load the PSFs
    unpadded_psfs, indices = csv_psfs.load_from_dir_index(psf_directory)

    # flip and transpose PSFs to undo what Zemax does:
    #unpadded_psfs = np.transpose(np.flip(unpadded_psfs, (0,1)), axes=(1,0,2))
    unpadded_psfs = np.transpose(unpadded_psfs, axes=(1,0,2))

    # load the metaman:
    metaman = load_PSFs.MetaMan(psf_meta_path)

    # get the dictionary of field origins once to save computation
    field_origins = metaman.field_origins
    shifts = metaman.shifts

    # We're assuming that the pixel-dimensions of the object image will actually be the same as img_dims
    find_pixel_on_obj = lambda x,y: (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))
    # we care about the difference between the placement of the origin and the shift position
    #origins_pixel = {k: find_pixel_on_obj(*v) for k, v in metaman.field_origins}
    origins_pixel = {k: find_pixel_on_obj(*v) for k, v in metaman.field_origins.items()}
    shift_from_origin = {}
    for index in indices:
        shift_from_origin[index] = (shifts[index][0] - origins_pixel[index][0], shifts[index][1] - origins_pixel[index][1])
        # since everything given by metaman is [x,y] we need to reverse to match numpy and linear algebra
        #shift_from_origin[index] = (shifts[index][1] - origins_pixel[index][1], shifts[index][0] - origins_pixel[index][0])

    # dictionary comprehension for what's commented out above
    #shift_from_origin = {index: (shifts[index][0] - origins_pixel[index][0], shifts[index][1] - origins_pixel[index][1]) for index in indices}

    # pad the PSFs based on the difference between the the calculated origin pixel of the field
    # and the shift of the PSF, all from the metaman
    padded_psfs = []
    for psf_index in range(unpadded_psfs.shape[-1]):
        padded_psfs.append(load_PSFs.pad_to_position(unpadded_psfs[:,:,psf_index], shift_from_origin[indices[psf_index]], img_dims))

    psfs_ndarray = np.transpose(np.asarray(padded_psfs), (1,2,0))

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    rank = 28
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    comps, weights_interp=svm.calc_svd_indexed(psfs_ndarray, origins_pixel, indices, rank)
    weights_norm = np.absolute(np.sum(weights_interp[weights_interp.shape[0]//2-1,weights_interp.shape[1]//2-1,:],0).max())
    weights = weights_interp/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=comps/np.linalg.norm(np.ravel(comps))

    return h, weights

def generate_and_save(psf_directory, psf_meta_path, img_dims, obj_dims, h_save_path, weights_save_path):
    h, weights = generate_model(psf_directory, psf_meta_path, img_dims, obj_dims)
    save_generated(h, weights, h_save_path, weights_save_path)

def save_generated(h, weights, h_save_path, weights_save_path):
    h_dict = {"array_out": h}
    scipy.io.savemat(h_save_path, h_dict)
    weights_dict = {"array_out": weights}
    scipy.io.savemat(weights_save_path, weights_dict)

# TODO: for warp-convolve architecture, whereby we first multiply pixels by weights, then warp the resulting image in spatial domain,
# then convolve the warped image with the PSFs in the Fourier domain, we need some stuff:
# we need to produce, save and later load an entire shifts array, which is 3-dimensional: e.g. shape=(800, 1280, 2) where the last dimension
# separates the positions to which to shift this point in the X-direction and Y-direction.
# Also, need a version of the above-given one-shot SVD methods for generation and saving with only center-padding.
# Also, need an altered version of the simulator which implements this warp-convolve algorithm.
