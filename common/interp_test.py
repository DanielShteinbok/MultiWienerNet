import one_shot_svd
from scipy.sparse.linalg import svds
import numpy as np
import csv_psfs
from scipy.interpolate import griddata

import sys
sys.path.append("../tensorflow")
import load_PSFs

# TODO: import everything as per one_shot_svd and manage dependencies
#import 

def find_pixel_on_obj(x, y, img_dims, obj_dims):
    """
    It generally makes sense that the object space is separate from the image space,
    so it makes sense that they can be in different units in general and can be represented
    by matrices with different dimensions. This is how everything is presented in other functions.

    However, when we want to deal with images, we want to deal with a finite number of pixels.
    It is convenient for this to be the same number of pixels in both the object and the image.

    If that assumption is true (both images have the same pixel dimensions), then it is possible
    to relate a point in the object space to a pixel in the object space by this function.

    Parameters:
        x: the x-coordinate in object space
        y: the y-coordinate in object space
        obj_dims: the object field-of-view, as (y, x)
        img_dims: the pixel dimensions of the image, as (y, x)
    """
    return (int(x*img_dims[1]/obj_dims[1]), int(y*img_dims[0]/obj_dims[0]))


def calc_svd_nointerp(yi_reg,si,rnk, imgdims, method='nearest'):
    """
    Directly the first half of svd_model.calc_svd
    """
    [Ny, Nx] = yi_reg[:,:,0].shape;
    print('creating matrix\n')
    Mgood = yi_reg.shape[2];
    ymat = np.zeros((Ny*Nx,Mgood));
    # NOTE this is where I suspect the problem of ordering happens
    # the original order of the ymat SHOULD be kept; it's the product of the weights and h
    # that matters at the end of the day, and that product should be the same
    ymat=yi_reg.reshape(( yi_reg.shape[0]* yi_reg.shape[1], yi_reg.shape[2]),order='F')

    print('done\n')

    print('starting svd...\n')

    print('check values of ymat')
    [u,s,v] = svds(ymat,rnk);


    comps = np.reshape(u,[Ny, Nx,rnk],order='F');
    vt = v*1
    # s=np.flip(s)
    # vt=np.flipud(vt)
    weights = np.zeros((Mgood,rnk));
    for m  in range (Mgood):
        for r in range(rnk):
            weights[m,r]=s[r]*vt[r,m]

    return comps, weights
    #return np.flip(comps,-1), np.flip(weights,-1)


# We don't need a metaman because we don't do any interpolation or any shifting here
def generate_unpadded_nointerp(psf_directory, img_dims, obj_dims, method="nearest"):
    """
    Generate the h and weights, where weights is NOT interpolated
    (for K PSFs, contains K entries)
    otherwise should be identical to one_shot_svd.generate_unpadded
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

    # don't need this because we won't be interpolating anything yet
    #origins_pixel = {k: find_pixel_on_obj(*v, img_dims, obj_dims) for k, v in metaman.field_origins.items()}

    # perform the SVD and interpolate weights based on those pixel values that we calculated
    #rank = 28
    rank = unpadded_psfs.shape[2] - 1
    # BELOW is the problem: PSF-origins_pixel mismatch
    #comps, weights_interp=svm.calc_svd(psfs_ndarray,origins_pixel,rank)
    #comps, weights_interp=svm.calc_svd_indexed_sized(unpadded_psfs, origins_pixel, indices, rank, img_dims, method=method)
    comps, weights_flattened=calc_svd_nointerp(
            unpadded_psfs, indices, rank, img_dims)

    # TODO reshape weights appropriately into (height, width, rank) form
    # since calc_svd_nointerp starts by reshaping the yi_reg FORTRAN-style,
    # we must keep with the fashion
    #weights_reshaped = np.reshape(weights_flattened, unpadded_psfs.shape, order='F')
    # NOTE actually, weights_flattened is of shape (N, k) for kth rank of SVD and N PSFs
    #weights_norm = np.absolute(np.sum(
        #weights_reshaped[weights_reshaped.shape[0]//2-1,weights_reshaped.shape[1]//2-1,:],0
        #).max())
    #weights = weights_reshaped/weights_norm;
    # no reason to do anything to weights; they aren't by pixel, but rather by 
    # out-of-order PSF
    weights = weights_flattened.transpose((1,0))

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=comps/np.linalg.norm(np.ravel(comps))
    # no reason to normalize h

    return h, weights, indices


def interped_kernels(weights, indices, metaman, img_dims, obj_dims, method="linear"):
    """
    Take the weights, go through each and try to interpolate its
    value as if it weren't there
    """
    # TODO: make si from the metaman's shifts

    xi=[]
    yi=[]
    # si_list passed directly as ordered list
    #si_list=list(si.values())
    si = {k: find_pixel_on_obj(*v, img_dims, obj_dims) 
            for k, v in metaman.field_origins.items()}
    for index in indices:
        xi.append(si[index][0])
        yi.append(si[index][1])

    # TODO we need to make ndarrays from each of xi, yi
    # so that we can take advantage of slicing
    # (or maybe we can do the latter without the former??)
    xi_ndarray = np.asarray(xi)
    yi_ndarray = np.asarray(yi)

    # weights_interp should be of the same shape as weights
    # then we go through each column thereof, keeping the index j
    # we find the origin of the jth psf through xi[j]
    # we then take the slice of xi excluding the jth element,
    # we take the slice of yi excluding the jth element,
    # we take the slice of weights excluding the jth column,
    # we interpolate the value of the jth PSF,
    # we store that in the appropriate column of the matrix that we've created for the purpose.
    weights_interp = np.empty_like(weights)
    for j in range(weights.shape[1]):
        # the point at which we are trying to interpolate
        # that is, the location of the origin of the jth PSF
        # transposed because numpy wants shape (m, D) where m=1 and D=2
        # NOTE PSFs are 1-indexed (because Fields are 1-indexed by Zemax)
        # so we need to add 1 to the index of si
        interp_coord = np.asarray([si[j+1]])

        # mask to select all but the jth element
        # should be [True, ..., True, False, True, ..., True]
        # where the only False is the jth entry
        # and the length of the whole thing is the same as the number of PSFs
        mask = np.arange(len(si)) != j
        for r in range(weights.shape[0]):
            weights_interp[r,j] = griddata((xi_ndarray[mask],yi_ndarray[mask]),
                    (weights[r,:])[mask],interp_coord,method=method)

    return weights_interp
