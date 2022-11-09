import one_shot_svd
import numpy as np
import scipy
import scipy.sparse

import load_PSFs
import tempfile

def mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, ker_dims, rows_disk, cols_disk, vals_disk, quite_small=0.001):
    """
    create the shifted COO mastermat.
    This can later be gone through to make the CSR that we need for the actual simulations.
    Parameters:
        kermat: the b-by-k CSR matrix of vectorized kernels
        weightsmat: the dense k-by-p matrix of pixel weights
        shifts: the p-vector of modulo-encoded shifts
        img_dims: (height, width) dimensions of the image
        ker_dims: (height, width) dimensions of the kernel; e.g. both sqrt of b
        rows_disk: (huge,) containing the row indices
        cols_disk: (huge,) containing the col indices
        vals_disk: (huge,) containing the values for our COO matrix
    """
    # extract only the pixels in the field of view
    # it's expected that everything which is to be shifted off the FOV in the x-dimension
    # has already had its shift set to a large negative value, since otherwise we can't distinguish
    # diagonal shifting down and to the left from x-overflow
    #selector = [shifts >= 0 and shifts < (img_dims[0]+1)*img_dims[1]]
    #pixels_in_fov = weightsmat.swapaxes(0,1)[selector]
    # shift to convert kernel space to image space
    #convert_shift = np.ravel((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*img_dims[1]).swapaxes(0,1))
    # should be adding img_dims[1] - ker_dims[1] for each row, because otherwise convert_shift basically causes
    # a lateral shift with each line
    convert_shift = np.ravel((np.ones((ker_dims[1], ker_dims[0]))*np.arange(ker_dims[0])*(img_dims[1]-ker_dims[1])).swapaxes(0,1))

    # simply skipping columns will lead to virtual zero-columns in the mastermat

    # shifts is going to have, e.g. shape=(2,1024000)
    not_nan_selector = (~np.isnan(shifts[:,0]))*(~np.isnan(shifts[:,1]))
    relevant_pixels = np.arange(weightsmat.shape[1])[not_nan_selector]

    add_index = 0
    #for pixel_ind in range(weightsmat.shape[1]):
    for pixel_ind in relevant_pixels:
        # a print statement to let us know where we are
        print("pixel_ind: ", pixel_ind)
        # the specific PFS vector produced by matrix muliplication of weights vector with
        # kernel matrix
        out_col = csr_kermat.dot(weightsmat[:,pixel_ind])

        # the position in which the top-left corner of the kernel must end up after shifting,
        # in vector form (top-left-corner shift):
        #tlcs = int(shifts[pixel_ind]) \
                #- ker_dims[1]*(ker_dims[0]//2-1) - (ker_dims[1]//2 - 1) \

        # get the indices of all nonzero values in the out_col b-vector
        #kernel_row_ind = np.arange(out_col.shape[0]).reshape(out_col.shape[0],1)[out_col>quite_small]
        # make kernel_row_ind all indices, then select by value later
        kernel_row_ind = np.arange(out_col.shape[0]).reshape(out_col.shape[0],1)
        value_selector = (out_col > quite_small).reshape((out_col.shape[0],1))

        # select only those indices that don't get cut off the FOV
        #kernel_row_ind_keep = kernel_row_ind[\
                #(kernel_row_ind % ker_dims[1] + tlcs % img_dims[1] < img_dims[1]) \
                #*(kernel_row_ind % ker_dims[1] + tlcs % img_dims[1] > 0)\
                #*(kernel_row_ind + tlcs < img_dims[0]*img_dims[1])\
                #*(kernel_row_ind + tlcs > 0)]

        # select those indices that don't get clipped off, using the new approach:
        shift_selector = ((kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] < img_dims[1]) # within right side
        * (kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] > 0) # within left side
        * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] < img_dims[0]) # above bottom
        * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] > 0) # below top
        )

        selector = shift_selector*value_selector

        kernel_row_ind_keep = kernel_row_ind[selector]

        if kernel_row_ind_keep.shape[0] > 0:
            print("keeping something")

        # for the above-selected unclipped indices, select only the corresponding values
        values = out_col[kernel_row_ind_keep]

        # convert_shift excluding small values
        #wo_zero = convert_shift[out_col>quite_small]
        # wo_zero excluding clipped
        #convert_keep = wo_zero[selector[:,0]]
        #convert_keep = convert_shift[selector[:,0]]
        convert_keep = convert_shift[kernel_row_ind_keep]

        # shift the the row indices as needed:
        img_row_ind = kernel_row_ind_keep + convert_keep + int(shifts[pixel_ind, 0])*img_dims[1] + shifts[pixel_ind,1]

        # add the row index to the row indices-row of the out_array memmap
        #out_array[0, add_index:img_row_ind.shape[0] + add_index] = img_row_ind[:] # row indices
        rows_disk[add_index:img_row_ind.shape[0] + add_index] = img_row_ind[:] # row indices
        #out_array[1, add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        cols_disk[add_index:img_row_ind.shape[0] + add_index] = pixel_ind*np.ones(img_row_ind.shape[0]) # column indices
        #out_array[2, add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        vals_disk[add_index:img_row_ind.shape[0] + add_index] = values[:] # values
        #out_array.flush()
        rows_disk.flush()
        cols_disk.flush()
        vals_disk.flush()
        add_index = add_index + img_row_ind.shape[0]

def mastermat_coo_creation_logic_fast(csr_kermat, weightsmat, shifts, img_dims, ker_dims, rows_disk, cols_disk, vals_disk, quite_small=0.001):
    """
    A faster, but MUCH more memory-heavy approach to making the mastermat coo matrix.
    the same parameters as mastermat_coo_creation_logic except huge_mat_disk
    huge_mat_disk is a B-by-P matrix, where B is the number of pixels in the kernel and  P in the image.

    DO NOT USE YET
    """
    huge_mat = csr_kermat.dot(weightsmat)
    huge_mat_coo = scipy.sparse.coo_matrix(huge_mat)
    # TODO: could assign elements to COO column-by-column
    # could consider shifts through meshgrid

    # TODO: perform shifts, assign to rows_disk and cols_disk
    # First, want to identify clipped edges as before
    #shift_selector = ((kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] < img_dims[1]) # within right side
    #    * (kernel_row_ind % ker_dims[1] + shifts[pixel_ind, 1] > 0) # within left side
    #    * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] < img_dims[0]) # above bottom
    #    * (kernel_row_ind // ker_dims[1] + shifts[pixel_ind, 0] > 0) # below top
    #    )

    # shift selector should have the same dimensions as huge_mat_coo
    # need to somehow get the x- and y- indices of the elements in huge_mat_coo in such a way that we can work
    # with it conveniently with numpy
    # Rather than going the meshgrid direction, we could just work with entries in huge_mat_coo.row and huge_mat_coo.col
    # 
    #selector = huge_mat_coo.row huge_mat_coo.col

def make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims, 
        savepath = ("row_inds_csr.npy", "col_inds_csr.npy", "values_csr.npy"), w_interp_method="nearest", s_interp_coords="cartesian"):
    """
    Complete process for making the mastermat and converting to CSR form,
    then saving as .npy to the given path.
    """
    metaman = load_PSFs.MetaMan(psf_meta_path)
    h, weights = one_shot_svd.generate_unpadded(psfs_directory, metaman, img_dims, obj_dims, method=w_interp_method)

    # get the shifts to apply to each point
    if s_interp_coords=="cartesian":
        shifts = one_shot_svd.interpolate_shifts(metaman, img_dims, obj_dims)
    elif s_interp_coords=="circular":
        shifts = one_shot_svd.interpolate_shifts_circular(metaman, img_dims, obj_dims)

    # the reshaped matrix of PSFs, where each column is a vectorized PSF
    #kermat = psfs.reshape((psfs.shape[0]*psfs.shape[1], psfs.shape[2]))
    # reshape tries to change the last axis first, so we need to transpose the matrix
    # to put the x- and y- axes at the end and 
    kermat = h.transpose((2, 0, 1)) \
    .reshape((h.shape[2], h.shape[0]*h.shape[1])) \
    .transpose((1,0)) # don't want to transpose, because we want k by \beta matrix

    # the reshaped matrix of weights
    # the column is the pixel index, the row is the kernel index
    weightsmat = weights.transpose((2,0,1)) \
    .reshape((weights.shape[2], weights.shape[0]*weights.shape[1]))

    # compressed sparse row matrix version of kermat 
    # that we will henceforth use for multiplication
    csr_kermat = scipy.sparse.csr_matrix(kermat)
    
    # these files should be entirely temporary
    # TODO make it so, could use tempfile module, including tempfile.mkdtemp

    with tempfile.TemporaryDirectory() as name:
        prefix = name + "/"
        row_inds = np.memmap(prefix + 'row_inds_temp.dat', mode='w+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
        col_inds = np.memmap(prefix + 'col_inds_temp.dat', mode='w+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
        values = np.memmap(prefix + 'values_temp.dat', mode='w+', shape=(100*img_dims[0]*img_dims[1]), dtype=np.float64)

        mastermat_coo_creation_logic(csr_kermat, weightsmat, shifts, img_dims, h.shape, row_inds, col_inds, values)

        NNZ = values[values!=0].shape[0] # the number of nonzero values

        row_inds_csr = np.memmap(prefix + 'row_inds_temp_csr.dat', mode='w+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
        col_inds_csr = np.memmap(prefix + 'col_inds_temp_csr.dat', mode='w+', shape=(NNZ), dtype=np.uint64)
        values_csr = np.memmap(prefix + 'values_temp_csr.dat', mode='w+', shape=(NNZ), dtype=np.float64)
        
        row_inds_csr, col_inds_csr, values_csr = compute_csr(row_inds, col_inds, values)

        np.save(savepath[0], row_inds_csr)
        np.save(savepath[1], col_inds_csr)
        np.save(savepath[2], values_csr)

def load_memmaps(img_dims, coo_paths=('row_inds.dat', 'col_inds.dat', 'values.dat'), csr_paths=('row_inds_csr.dat', 'col_inds_csr.dat', 'values_csr.dat')):
    row_inds = np.memmap(coo_paths[0], mode='r', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    col_inds = np.memmap(coo_paths[1], mode='r', shape=(100*img_dims[0]*img_dims[1]), dtype=np.uint64)
    values = np.memmap(coo_paths[2], mode='r', shape=(100*img_dims[0]*img_dims[1]), dtype=np.float64)

    #NNZ = values[values!=0].shape[0] # the number of nonzero values

    #row_inds_csr = np.memmap(csr_paths[0], mode='r+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
    #col_inds_csr = np.memmap(csr_paths[1], mode='r+', shape=(NNZ), dtype=np.uint64)
    #values_csr = np.memmap(csr_paths[2], mode='r+', shape=(NNZ), dtype=np.float64)

    row_inds_csr, col_inds_csr, values_csr = load_csr_memmaps(row_inds, col_inds, values, img_dims, csr_paths=('row_inds_csr.dat', 'col_inds_csr.dat', 'values_csr.dat'))

    return row_inds_csr, col_inds_csr, values_csr, row_inds, col_inds, values

def load_csr_memmaps(row_inds, col_inds, values, img_dims, csr_paths=('row_inds_csr.dat', 'col_inds_csr.dat', 'values_csr.dat')):
    NNZ = values[values!=0].shape[0] # the number of nonzero values

    row_inds_csr = np.memmap(csr_paths[0], mode='r+', shape=(img_dims[0]*img_dims[1] + 1), dtype=np.uint64)
    col_inds_csr = np.memmap(csr_paths[1], mode='r+', shape=(NNZ), dtype=np.uint64)
    values_csr = np.memmap(csr_paths[2], mode='r+', shape=(NNZ), dtype=np.float64)

    return row_inds_csr, col_inds_csr, values_csr

def compute_csr(row_inds, col_inds, values):
    """
    just a wrapper around scipy's sparse.csr_matrix
    which returns the three ndarrays of underlying data behind the computed CSR
    so that they can be saved to disk
    """
    out_csr = scipy.sparse.csr_matrix((values, (row_inds, col_inds)))
    return out_csr.indptr, out_csr.indices, out_csr.data

def simulate_image(img, csr_mastermat):
    """
    ravels the image, matrix-multiplies with the mastermat, unravels the result
    """
    img_vec = img.reshape((img.shape[0]*img.shape[1]))
    return csr_mastermat.dot(img_vec).reshape(img.shape)

def load_csr_files(csr_paths=('row_inds_csr.npy', 'col_inds_csr.npy', 'values_csr.npy')):
    row_inds_csr = np.load(csr_paths[0])
    col_inds_csr = np.load(csr_paths[1])
    values_csr = np.load(csr_paths[2])

    return row_inds_csr, col_inds_csr, values_csr


