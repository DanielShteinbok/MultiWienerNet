import one_shot_svd
import numpy as np

#row_inds_csr, col_inds_csr, values_csr = one_shot_svd.make_mastermat("/home/dshteinbok/nV3_PSFs_flat", "/home/dshteinbok/nV3_PSFs_flat_meta/metafile.csv", (800,1280), (640,1024))

#row_inds_csr, col_inds_csr, values_csr, row_inds, col_inds, values = one_shot_svd.load_memmaps((800,1280))

#row_inds, col_inds, values = one_shot_svd.make_mastermat_coo("/home/dshteinbok/nV3_PSFs_flat", "/home/dshteinbok/nV3_PSFs_flat_meta/metafile.csv", (800,1280), (640,1024))

#row_inds_csr, col_inds_csr, values_csr = one_shot_svd.load_csr_memmaps(row_inds, col_inds, values, (800,1280))
row_inds_csr, col_inds_csr, values_csr, row_inds, col_inds, values = one_shot_svd.load_memmaps((800,1280))

one_shot_svd.compute_csr(row_inds_csr, col_inds_csr, values_csr, row_inds, col_inds, values)

np.save("row_inds_csr.npy", row_inds_csr)
np.save("col_inds_csr.npy", col_inds_csr)
np.save("values_csr.npy", values_csr)
