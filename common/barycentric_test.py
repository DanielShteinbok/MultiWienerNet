import numpy as np
import scipy.interpolate
import one_shot_svd

# generate 100 values as if in a square
values = np.arange(100)

# generate the indices for these values
# we shift them by 0.2 to the right and bottom
[inds_x_shaped, inds_y_shaped] = np.meshgrid(np.arange(10), np.arange(10))
inds_x = np.ravel(inds_x_shaped) + 11.2
inds_y = np.ravel(inds_y_shaped) + 11.2

# we need a (100, 2) ndarray for meshgrid
points_2d = np.empty((inds_x.shape[0], 2))
points_2d[:,0] = inds_x
points_2d[:,1] = inds_y

points_complex = inds_x + 1j*inds_y

# points to interpolate at
[xix_shaped, xiy_shaped] = np.meshgrid(np.arange(32), np.arange(32))
xix = np.ravel(xix_shaped)
xiy = np.ravel(xiy_shaped)

xi = np.empty((xix.shape[0], 2))
xi[:,0] = xix
xi[:,1] = xiy

xic = xix + 1j*xiy

# perform the interpolation with scipy
scipy_interp = scipy.interpolate.griddata(points_2d, values, xi).reshape((32,32))
my_interp = one_shot_svd.interp_grid(points_complex, values)

# moment of truth!
print(np.allclose(scipy_interp[~np.isnan(scipy_interp)], my_interp[~np.isnan(scipy_interp)]))
