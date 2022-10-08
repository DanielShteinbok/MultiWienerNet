import glob
import numpy as np

def load_from_dir(psfs_path):
    """
    load a stack of PSFs as all the CSV files from the specified directory
    
    psfs_path: the path to the directory with all the CSV files
    
    Returns: np.ndarray of shape (width, height, number_of_psfs)
    """
    psf_paths = glob.glob(psfs_path.removesuffix('/') + '/*')
    # iterate through that list,
    # open and append each to the psfs array,
    psfs = []
    for path in psf_paths:
        psfs.append(np.loadtxt(path, delimiter=',', encoding='utf-8-sig'))
    # convert the psfs array to an np.ndarray
    psfs = np.transpose(np.asarray(psfs), (1,2,0))
    return psfs

def pad_as_center(initial_psfs, height, width):
    '''
    zero-pad a stack of PSFs to the desired width and height, so they stay in the center
    '''
    hdiff = height - initial_psfs.shape[0]
    wdiff = width - initial_psfs.shape[1]
    
    if hdiff < 0 or wdiff < 0:
        raise ValueError("initial PSF larger than expected size")
    # Pad PSF with zeros to the specified height and width so that it ends up in the middle
    elif hdiff > 0 or wdiff > 0:
        initial_psfs = np.pad(initial_psfs,
                              ((np.math.ceil(hdiff/2), np.math.floor(hdiff/2)),
                              (np.math.ceil(wdiff/2), np.math.floor(wdiff/2)),
                              (0,0)))
    return initial_psfs