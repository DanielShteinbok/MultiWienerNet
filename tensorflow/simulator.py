# needed for ImageSimulator
#import tensorflow as tf
import forward_model_tf as fm
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2

import os

# experimentation with timing
import time

# class that allows you to load weights upon instantiation, then just pass images in and get images out
class ImageSimulator:
    def __init__(self, h_path, weights_path, input_4d=False):
        if input_4d:
            self.H, self.weights, self.crop_indices = fm.load_weights(h_path=h_path, weights_path=weights_path)
        else:
            self.H, self.weights, self.crop_indices = fm.load_weights_2d(h_path=h_path, weights_path=weights_path)
    def simulate(self, objectPath, my_simulation=True):
        '''
        objectPath: str path to the image to be simulated.
        
        Returns: obj, sim
            obj: image loaded from objectPath
            sim: simulated image
        '''
        last_time = time.time_ns() # TIMING
        
        im_in=imageio.v2.imread(objectPath)

        print(f"{'Read image in:' : <10}{time.time_ns()-last_time : >20}") # TIMING

        last_time = time.time_ns() # TIMING
        
        # TODO only resize if necessary
        im=cv2.resize(im_in,(self.weights.shape[1],self.weights.shape[0]))

        print(f"{'Resized image in:' : <10}{time.time_ns()-last_time : >20}") # TIMING
            
        last_time = time.time_ns() # TIMING
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

        print(f"{'Converted to grayscalse in:' : <10}{time.time_ns()-last_time : >20}") # TIMING

        last_time = time.time_ns() # TIMING
        # im is of type np.ndarray
        if my_simulation:
            sim = fm.sim_data(im,self.H,self.weights,self.crop_indices, a_svd_func=fm.my_A_2d_svd)
            print(f"{'Simulated in:' : <10}{time.time_ns()-last_time : >20}") # TIMING
            return im, sim
            #return im, fm.sim_data(im,self.H,self.weights,self.crop_indices, a_svd_func=fm.my_A_2d_svd)

        print(f"{'Simulated in:' : <10}{time.time_ns()-last_time : >20}") # TIMING
        sim = fm.sim_data(im,self.H,self.weights,self.crop_indices)
        return im, sim
        #return im, fm.sim_data(im,self.H,self.weights,self.crop_indices)
        
    def simulate_matrix(self, im_in, my_simulation=True, add_noise=True):
        last_time = time.time_ns() # TIMING
        
        # TODO only resize if necessary
        im=cv2.resize(im_in,(self.weights.shape[1],self.weights.shape[0]))

        print(f"{'Resized image in:' : <10}{time.time_ns()-last_time : >20}") # TIMING
            
        last_time = time.time_ns() # TIMING
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

        print(f"{'Converted to grayscalse in:' : <10}{time.time_ns()-last_time : >20}") # TIMING

        last_time = time.time_ns() # TIMING
        # im is of type np.ndarray
        if my_simulation:
            sim = fm.sim_data(im,self.H,self.weights,self.crop_indices, a_svd_func=fm.my_A_2d_svd)
            print(f"{'Simulated in:' : <10}{time.time_ns()-last_time : >20}") # TIMING
            return im, sim
            #return im, fm.sim_data(im,self.H,self.weights,self.crop_indices, a_svd_func=fm.my_A_2d_svd)

        print(f"{'Simulated in:' : <10}{time.time_ns()-last_time : >20}") # TIMING
        sim = fm.sim_data(im,self.H,self.weights,self.crop_indices,add_noise=add_noise)
        return im, sim
        #return im, fm.sim_data(im,self.H,self.weights,self.crop_indices)

def sim_from_pathlist(h_path, weights_path, pathlist, output_dir_simmed, output_dir_resized=None):
    '''
    pathlist: list of str, with each element being a path to a file
    output_dir: str, the directory into which to dump the simulated files
    '''
    # timing each step
    last_time = time.time_ns()
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir_simmed):
        os.makedirs(output_dir_simmed)
    
    print("created appropriate directories in:\t\t", time.time_ns()-last_time) # TIMING
    last_time = time.time_ns()
    
    # create the simulator
    simulator = ImageSimulator(h_path, weights_path)
    
    print("initialized simulator in:\t\t", time.time_ns()-last_time)
    
    # go through each path in pathlist, open the image, simulate it, and 
    for path in pathlist:
        last_time = time.time_ns()
        #im, sim = simulator.simulate(path)
        im, sim = simulator.simulate(path, my_simulation=False)
        
        print("simulated " + path + " in:\t\t", time.time_ns()-last_time) # TIMING
        last_time = time.time_ns() # TIMING
        
        # FIXME: imwrite automatically converts pixels from float64 to uint8, which may result in data loss.
        imageio.imwrite(output_dir_simmed.removesuffix('/') + '/' + os.path.basename(path), sim)
        
        print("finished saving simulated image in:\t\t", time.time_ns()-last_time) # TIMING
        last_time = time.time_ns() # TIMING
        
        if not output_dir_resized is None:
            imageio.imwrite(output_dir_resized.removesuffix('/') + '/' + os.path.basename(path), im)
            print("saved resized in:\t\t", time.time_ns()-last_time)
