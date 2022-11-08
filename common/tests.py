import unittest
import numpy.testing

import numpy as np

import svd_model

class CalcSVDTests(unittest.TestCase):
    # Overall, calc_svd seems kosher.
    @classmethod
    def setUpClass(cls):
        cls.ker_dims = (16, 32)
        cls.img_dims = (18, 34)
        cls.N = 8
        cls.rank = 7

        # fake PSFs which are NOT  normalized in any way
        cls.original_psfs = np.arange(cls.ker_dims[0]*cls.ker_dims[1]*cls.N, dtype=np.float) \
                .reshape((cls.ker_dims[0], cls.ker_dims[1], cls.N))

        # some made up points spaced evenly over the image, which represent
        # field origins for respective layers of the stack of psfs
        # has the form: {key (int) : value ([x, y])}
        # you should be able to feed this as si to calc_svd
        cls.original_origins = {}
        dx = cls.img_dims[1]/5
        dy = cls.img_dims[0]/3
        for i in range(cls.N):
            # will just put 2 rows of 4 for these particular dimensions
            #cls.original_origins[i + 1] = [(i%4 + 1)*dx, (i//4 + 1)*dy]
            cls.original_origins[i] = [(i%4 + 1)*dx - cls.img_dims[1]/2, 
                    (i//4 + 1)*dy - cls.img_dims[0]/2]

        # now we need scrambled origins for the indexed
        # for now, will take order: 2, 1, 4, 3, 6, 5, 8, 7
        #cls.scramble_order = [2, 1, 4, 3, 6, 5, 8, 7]
        cls.scramble_order = [2, 1, 4, 3, 6, 5, 0, 7]

        cls.scrambled_psfs = 1*cls.original_psfs
        
        # scrambled_origins should be the same as original_origins,
        # but the order of the keys would be different
        cls.scrambled_origins = {}
        for i in cls.scramble_order:
            cls.scrambled_origins[i] = cls.original_origins[i]

        for i in range(len(cls.scramble_order)):
            #print(i)
            #cls.scrambled_psfs[:,:,i] = cls.original_psfs[:,:,cls.scramble_order[i]-1]
            cls.scrambled_psfs[:,:,i] = cls.original_psfs[:,:,cls.scramble_order[i]]

    def test_indexed_vs_original(self):
        # NOTE this case only passes because (0,0) lies on the diagonal--
        # actually, calc_svd_indexed_sized produces the transpose of calc_svd

        # ensure that calc_svd_indexed can unscramble
        # PSF info correctly and agrees with calc_svd

        # generate pretend PSFs for calc_svd:
        # For now, our "PSFs" will be of shape (16, 32, 8)
        h1, w1 = svd_model.calc_svd(self.original_psfs, self.original_origins, self.rank)
        h2, w2 = svd_model.calc_svd_indexed_sized(self.scrambled_psfs, self.scrambled_origins, 
                self.scramble_order, self.rank, self.ker_dims)
        #self.assertEqual(np.any(h1-h2), False)

        # the matrix product of kernels and weights should be the same,
        # but the order of kernels and order of weights are each different between
        # the two versions of calc_svd being compared

        #h2_reordered = h2.transpose((2,0,1))[self.scramble_order].transpose((1, 2, 0))
        # urc is upper-right corner
        urc2 = h2*w2[0,0,:]
        urc1 = h1*w1[0,0,:]
        #self.assertEqual()
        #np.testing.assert_allclose(h1,h2_reordered)
        np.testing.assert_allclose(urc1, urc2)
        #self.assertEqual(np.any(w1-w2), False)
    def test_weights_shape(self):
        h, w = svd_model.calc_svd_indexed_sized(self.scrambled_psfs, self.scrambled_origins, 
                self.scramble_order, self.rank, self.img_dims)
        self.assertEqual(w.shape, (self.img_dims[0], self.img_dims[1], self.rank))

    # TODO: test whether a ones-vector left-multiplied by an unshifted mastermat
    # returns to you the sum of the PSFs
    def test_svd_fidelity(self):
        h, w = svd_model.calc_svd_indexed_sized(self.scrambled_psfs, self.scrambled_origins, 
                self.scramble_order, self.rank, self.ker_dims)
        # we want to test fidelity at the top left corner of the image.
        # That should be the first element in original_psfs
        urc = np.sum(h*w[0,0,:], -1)
        #print(urc.shape)

        #print(np.arange(urc.shape[0]*urc.shape[1]).reshape((urc.shape[0], urc.shape[1]))[urc != self.original_psfs[:,:,0]])

        np.testing.assert_allclose(urc, self.original_psfs[:,:,0], rtol=0.001, atol=1e-10)

# TODO: unit tests for mastermat_coo_logic
# to verify that the shifting works as expected.
if __name__ == '__main__':
        unittest.main()
