# needed for ImageSimulator
#import tensorflow as tf
import forward_model as fm
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2

import os

# class that allows you to load weights upon instantiation, then just pass images in and get images out
class ImageSimulator:
    def __init__(self, h_path, weights_path):
        self.H, self.weights, self.crop_indices = fm.load_weights(h_path=h_path, weights_path=weights_path)
    def simulate(self, objectPath):
        '''
        objectPath: str path to the image to be simulated.
        
        Returns: obj, sim
            obj: image loaded from objectPath
            sim: simulated image
        '''
        im_in=imageio.v2.imread(objectPath)
        
        # TODO only resize if necessary
        im=cv2.resize(im_in,(648,486))
            
        # if there are multiple channels, average them and convert to grayscale
        if len(im.shape) > 2:
            if len(im.shape) == 3:
                # the image is 2D, with some number of channels
                print("image has " + str(im.shape[2]) + " channels. Converting to grayscale.")
                # set each "pixel" to the average intensity of the three channels
                im=np.sum(im,-1)/im.shape[2]
            else:
                # there are more than 3 dimensions to the image. something is wrong.
                raise ValueError("image passed is " + str(len(im.shape)) + "-dimensional?!?!")
        # im is of type np.ndarray
        return im, fm.sim_data(im,self.H,self.weights,self.crop_indices)

def sim_from_pathlist(h_path, weights_path, pathlist, output_dir):
    '''
    pathlist: list of str, with each element being a path to a file
    output_dir: str, the directory into which to dump the simulated files
    '''
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create the simulator
    simulator = ImageSimulator(h_path, weights_path)
    # go through each path in pathlist, open the image, simulate it, and 
    for path in pathlist:
        _, sim = simulator.simulate(path)
        # FIXME: imwrite automatically converts pixels from float64 to uint8, which may result in data loss.
        imageio.imwrite(output_dir.strip('/') + '/' + os.path.basename(path), sim)
