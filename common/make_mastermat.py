import mastermat

#psfs_directory = "../data/nV3_PSFs_flat_hd"
#psfs_directory = "../data/nV3_PSFs_curve"
#psfs_directory = "../data/nV3_PSFs_flat"
#psfs_directory = "../data/nV3_PSFs_probe"
#psfs_directory = "../data/nV3_PSFs_probe_mark"
#psfs_directory = "../data/nV3_PSFs_probe_mark_green"
#psfs_directory = "../data/nV3_PSFs_probe_depth1"
#psfs_directory = "../data/nV3_PSFs_probe_depth2"
#psfs_directory = "../data/nV3_PSFs_probe_depth3"
#psfs_directory = "../data/nV3_PSFs_tilted"
#psf_meta_path = "../data/nV3_PSFs_flat_meta/metafile_hd.csv"
#psf_meta_path = "../data/nV3_PSFs_meta/metafile.csv"
#psf_meta_path = "../data/nV3_PSFs_flat_meta/metafile.csv"
#psf_meta_path = "../data/nV3_PSFs_flat_meta/metafile_probe_mark.csv"
#psf_meta_path = "../data/nV3_PSFs_probe_depths/metafile_probe_depth1.csv"
#psf_meta_path = "../data/nV3_PSFs_probe_depths/metafile_probe_depth2.csv"
#psf_meta_path = "../data/nV3_PSFs_probe_depths/metafile_tilted.csv"
#psf_meta_path = "../data/nV3_PSFs_flat_meta/metafile_probe.csv"
img_dims = (800, 1280)
obj_dims = (640, 1024)

#mastermat.make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_lin_circ_csr.npy", "col_inds_lin_circ_csr.npy", "values_lin_circ_csr.npy"),
        #w_interp_method="linear", s_interp_coords="circular")

#mastermat.make_mastermat_save(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_lin_csr.npy", "col_inds_lin_csr.npy", "values_lin_csr.npy"),
        #w_interp_method="linear")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve_csr.npy", "col_inds_curve_csr.npy", "values_curve_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve2_csr.npy", "col_inds_curve2_csr.npy", "values_curve2_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_curve_nearest_csr.npy", "col_inds_curve_nearest_csr.npy", "values_curve_nearest_csr.npy"),
        #w_interp_method="nearest")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_flat_rot_csr.npy", "col_inds_flat_rot_csr.npy", "values_flat_rot_csr.npy"),
        #w_interp_method="cubic", rotate_psfs=True)

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_diy_csr.npy", "col_inds_diy_csr.npy", "values_diy_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_hd_nearest_csr.npy", "col_inds_hd_nearest_csr.npy", "values_hd_nearest_csr.npy"),
        #w_interp_method="nearest")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_hd_csr.npy", "col_inds_hd_csr.npy", "values_hd_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_fm3_csr.npy", "col_inds_fm3_csr.npy", "values_fm3_csr.npy"),
        #w_interp_method="nearest")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_csr.npy", "col_inds_probe_csr.npy", "values_probe_csr.npy"),
        #w_interp_method="cubic")

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_green_csr.npy", "col_inds_probe_green_csr.npy", "values_probe_green_csr.npy"),
        #w_interp_method="cubic", avg_nnz=1000)

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_green_undistorted_csr.npy", "col_inds_probe_green_undistorted_csr.npy", "values_probe_green_undistorted_csr.npy"),
        #w_interp_method="cubic", avg_nnz=1000, original_shift=True)

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_undistorted_depth2_csr.npy", "col_inds_probe_undistorted_depth2_csr.npy", "values_probe_undistorted_depth2_csr.npy"),
        #w_interp_method="cubic", avg_nnz=1000, original_shift=True)

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_undistorted_depth3_csr.npy", "col_inds_probe_undistorted_depth3_csr.npy", "values_probe_undistorted_depth3_csr.npy"),
        #w_interp_method="cubic", avg_nnz=2000, original_shift=True)

#mastermat.make_mastermat_save_homemade(psfs_directory, psf_meta_path, img_dims, obj_dims,
        #savepath = ("row_inds_probe_undistorted_tilted_csr.npy", "col_inds_probe_undistorted_tilted_csr.npy", "values_probe_undistorted_tilted_csr.npy"),
        #w_interp_method="cubic", avg_nnz=2000, original_shift=False)

#mastermat.make_mastermat_save_homemade("../data/nV3_PSFs_probe_mark_green",
                                       #"../data/nV3_PSFs_flat_meta/metafile_probe_mark.csv",
                                       #img_dims, obj_dims,
        #savepath = ("row_inds_probe_green_undistorted_corrected_csr.npy", "col_inds_probe_green_undistorted_corrected_csr.npy", "values_probe_green_undistorted_corrected_csr.npy"),
        #w_interp_method="cubic", avg_nnz=1000, original_shift=False)

#mastermat.make_mastermat_save_homemade("../data/PSFs_probe_flat_notnormalized",
                                       #"../data/notnormalized_metafile.csv",
                                       #img_dims, obj_dims,
        #savepath = ("../data/mastermat/row_inds_notnormalized.npy", "../data/mastermat/col_inds_notmormalized.npy", "../data/mastermat/values_notnormalized.npy"),
        #w_interp_method="cubic", avg_nnz=1000, original_shift=False, cols_in_memory=100, quite_small=0.001)

#mastermat.make_mastermat_save_homemade("../data/PSFs_probe_flat_normalized",
                                       #"../data/normalized_metafile.csv",
                                       #img_dims, obj_dims,
        #savepath = ("../data/mastermat/row_inds_normalized.npy", "../data/mastermat/col_inds_normalized.npy","../data/mastermat/values_normalized.npy"),
        #w_interp_method="cubic", avg_nnz=1000, original_shift=False, cols_in_memory=500, quite_small=0.001)


#mastermat.make_mastermat_save_homemade("../data/PSFs_probe_flat_ringed_green",
                                       #"../data/ringed_green_metafile.csv",
                                       #img_dims, obj_dims,
        #savepath = ("../data/mastermat/row_inds_ringed.npy", "../data/mastermat/col_inds_ringed.npy","../data/mastermat/values_ringed.npy"),
        #w_interp_method="cubic", avg_nnz=2000, original_shift=False, cols_in_memory=200, quite_small=0.001)

#mastermat.make_mastermat_save_homemade("../data/PSFs_curved_notnormalized",
                                       #"../data/curved_notnormalized_metafile.csv",
                                       #img_dims, obj_dims,
        #savepath = ("../data/mastermat/row_inds_curved.npy", "../data/mastermat/col_inds_curved.npy","../data/mastermat/values_curved.npy"),
        #w_interp_method="cubic", avg_nnz=2000, original_shift=False, cols_in_memory=200, quite_small=0.001)

#mastermat.make_mastermat_save_homemade("../data/PSFs_ringed_curved_notnormalized",
                                       #"../data/ringed_curved_notnormalized_metafile.csv",
                                       #img_dims, obj_dims,
        #savepath = ("../data/mastermat/row_inds_ringed_curved.npy", "../data/mastermat/col_inds_ringed_curved.npy","../data/mastermat/values_ringed_curved.npy"),
        #w_interp_method="cubic", avg_nnz=2000, original_shift=False, cols_in_memory=200, quite_small=0.001)

#mastermat.make_mastermat_save_homemade("../data/PSFs_ringed_curved_normalized",
                                       #"../data/ringed_curved_normalized_metafile.csv",
                                       #img_dims, obj_dims,
        #savepath = ("../data/mastermat/row_inds_ringed_curved_normalized.npy", "../data/mastermat/col_inds_ringed_curved_normalized.npy","../data/mastermat/values_ringed_curved_normalized.npy"),
        #w_interp_method="cubic", avg_nnz=2000, original_shift=False, cols_in_memory=50, quite_small=0.01)

mastermat.make_mastermat_save_homemade("../data/PSFs_Strehl",
                                       "../data/metafile_Strehl.csv",
                                       img_dims, obj_dims,
        savepath = ("../data/mastermat/row_inds_strehl.npy", "../data/mastermat/col_inds_strehl.npy","../data/mastermat/values_strehl.npy"),
        w_interp_method="cubic", avg_nnz=2000, original_shift=False, cols_in_memory=200, quite_small=0.00001, strehl_interp_method="cubic")


