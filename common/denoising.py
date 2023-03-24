import numpy as np
import scipy.ndimage
import cv2

# bm3d should be included with opencv,
# but due to licensing issues the latter must be built with a specific flag.
# Since opencv is provided by conda, I can't easily rebuild the library.
# Thus, I am importing a special library just for bm3d.
# This is downloaded through: pip install bm3d
import bm3d

def dff(image, background):
    return image - background - np.min(np.reshape(image - background, (background.shape[0]*background.shape[1], -1)))

def mse(a, b):
    return np.sum((a - b)**2)/a.shape[-1]/a.shape[-2]

def rescale_to_one(img):
    """
    Rescale all the pixel values in your image to range from 0 to 1.
    Should work for any shape 'img', even 3D
    """
    # DANGER: Watch out for overflows... or just do this with floating point img
    img_shifted = img.astype(np.float64) - np.min(img)
    # also, what happens with int type in line below???
    # for the above two paranoiae, I just made the thing into a float64
    return img_shifted/np.max(img_shifted)

def get_similar(full_stack, index, num_imgs, mse_threshold=None, print_mses=False):
    # this just holds the two "edges"
    other_inds = [index-1, index+1]

    # rather than keeping a stack of the frames we want, we'll just keep a list of their indices
    # this is more computationally efficient, but also allows for dynamically changing
    # the number of images we care about, e.g. due to thresholding
    out_stack_inds = [index]
    #out_stack = np.empty((num_imgs, full_stack.shape[1], full_stack.shape[2]))
    def add_img(i, side):
        # side = 0 if adding previous, side=1 if adding next
        # verify here that the mse is below the threshold
        mse_val = mse(full_stack[index], full_stack[other_inds[side], :, :])
        if mse_threshold is not None:
            if  mse_val > mse_threshold:
                # do nothing if mse is too high
                return
        if print_mses:
            print("On iteration " + str(i) + ", MSE is: " + str(mse_val))
        #out_stack[i, :, :] = full_stack[other_inds[side],:,:]
        out_stack_inds.append(other_inds[side])
        other_inds[side] += 2*side - 1
    for i in range(num_imgs - 1):
        if other_inds[1] >= full_stack.shape[0]:
            add_img(i, 0)
        elif other_inds[0] < 0:
            add_img(i, 1)
        elif mse(full_stack[other_inds[0],:,:], full_stack[index,:,:]) > mse(full_stack[other_inds[1],:,:], full_stack[index,:,:]):
            add_img(i, 1)
        else:
            add_img(i, 0)
    #return out_stack
    # this should return exactly the same way that it did before,
    # only now we've changed the internal behaviour above
    return full_stack[out_stack_inds, :, :]

def full_denoising(full_stack, index, background, num_images=5, h=4, verbose=False, denoising_method='NLM'):
    """
    Parameters:
        full_stack should be of shape (C, height, width) where height is the height of the image (e.g. 800)
        and width is the width of the image (e.g. 1280)

        h is the parameter passed to whichever algorithm you use. In the case of NLM, this is the h parameter.
        In the case of BM3D, this is the standard derivative of the noise.
        In the case of median filtering, this is actually a tuple! Namely, a tuple representing the size of
        the window within which to perform the filtering.
    """
    out_image = np.empty_like(full_stack[index, :,:], dtype=np.uint8)
    #cv2.fastNlMeansDenoising((dff(
        #np.mean(get_similar(full_stack, index, num_images), axis=0),
    #background)//2).astype(np.uint8), out_image, h=h)
    #background)//256).astype(np.uint8), out_image, h=h)
    dff_img = dff(
        np.mean(get_similar(full_stack, index, num_images, print_mses=verbose), axis=0),
    background)
    if denoising_method is None:
        out_image = (dff_img/dff_img.max()*255).astype(np.uint8)
    elif denoising_method == 'NLM':
        cv2.fastNlMeansDenoising((dff_img/dff_img.max()*255).astype(np.uint8), out_image, h=h)
    elif denoising_method == 'BM3D':
        #cv2.xphoto.bm3dDenoising((dff_img/dff_img.max()*255).astype(np.uint8), out_image, h=h)
        # in the case of bm3d, h will be the variance of the noise
        out_image = bm3d.bm3d(np.atleast_3d(dff_img/dff_img.max()*255).astype(np.uint8), h)
    elif denoising_method == 'median':
        # in this case, h should be a tuple. LOL.
        out_image = scipy.ndimage.median_filter(dff_img, size=h)
    #cv2.fastNlMeansDenoising((dff_img/2).astype(np.uint8), out_image, h=h)
    #background)//256).astype(np.uint8), out_image, h=h)
    return out_image
