import mastermat
import os
import imageio
import numpy as np

# experimentation with timing
import time

def sim_from_pathlist(mastermat_csr, pathlist, output_dir_simmed, noise_stdev=0):
    '''
    pathlist: list of str, with each element being a path to a file
    output_dir: str, the directory into which to dump the simulated files
    
    images contained in pathlist must already be the correct size for the mastermatrix
    '''
    # timing each step
    last_time = time.time_ns()
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir_simmed):
        os.makedirs(output_dir_simmed)
    
    print("created appropriate directories in:\t\t", time.time_ns()-last_time) # TIMING
    last_time = time.time_ns()
    
    print("initialized simulator in:\t\t", time.time_ns()-last_time)
    
    # go through each path in pathlist, open the image, simulate it, and 
    for path in pathlist:
        last_time = time.time_ns()
        #im, sim = simulator.simulate(path)
        
        sim = mastermat.simulate_image(imageio.v2.imread(path),mastermat_csr)
        
        if noise_stdev > 0:
            sim += np.random.normal(0, noise_stdev, size=sim.shape)
        
        print("simulated " + path + " in:\t\t", time.time_ns()-last_time) # TIMING
        last_time = time.time_ns() # TIMING
        
        # FIXME: imwrite automatically converts pixels from float64 to uint8, which may result in data loss.
        imageio.imwrite(output_dir_simmed.removesuffix('/') + '/' + os.path.basename(path), sim)
        
        print("finished saving simulated image in:\t\t", time.time_ns()-last_time) # TIMING
        last_time = time.time_ns() # TIMING