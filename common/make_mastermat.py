import mastermat

psfs_directory = "/home/dshteinbok/nV3_PSFs_flat"
psf_meta_path = "/home/dshteinbok/nV3_PSFs_flat_meta/metafile.csv"
img_dims = (800, 1280)
obj_dims = (640, 1024)

#mastermat.make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims, 
        #savepath = ("row_inds_lin_circ_csr.npy", "col_inds_lin_circ_csr.npy", "values_lin_circ_csr.npy"), 
        #w_interp_method="linear", s_interp_coords="circular")

mastermat.make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims, 
        savepath = ("row_inds_lin_csr.npy", "col_inds_lin_csr.npy", "values_lin_csr.npy"), 
        w_interp_method="linear")
