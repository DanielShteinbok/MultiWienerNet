import numpy as np
import os.path

def load_image(filename, index=0, dimensions=(400,640)):
    with open(filename, 'rb') as file:
        # get metadata size
#         file.seek(-8)
#         metadata_size = int(file.read(8))
#         file.seek(0)
#         all_data = file.read()
        file.seek(2*(dimensions[0]+8)*dimensions[1]*index)
        img_data = np.frombuffer(file.read(2*(dimensions[0]+4)*dimensions[1]), 
                                 dtype=np.uint16)[2*dimensions[1]:(dimensions[0]+2)*dimensions[1]].reshape(dimensions)
    return img_data

def load_image2(filename, index=0, dimensions=(400,640), metadata_size=(5120,5120)):
    with open(filename, 'rb') as file:
        # get metadata size
#         file.seek(-8)
#         metadata_size = int(file.read(8))
#         file.seek(0)
#         all_data = file.read()
        file.seek((2*dimensions[0]*dimensions[1] + metadata_size[0] + metadata_size[1])*index)
        img_data = np.frombuffer(file.read(2*dimensions[0]*dimensions[1] + metadata_size[0] + metadata_size[1]), 
                                 dtype=np.uint16)[metadata_size[0]//2:-metadata_size[1]//2].reshape(dimensions)
    return img_data

def mean_of_images(filename, dimensions=(400,640), metadata_size=(5120, 5120)):
    """
    get the mean of all the images in the stack.
    Goes until we can't load another full frame.
    Parameters:
        metadata_size: tuple (int before, int after) contains the metadata size in bytes before and after each image
    """
    with open(filename, 'rb') as file:
#         file.seek(metadata_size[0])
        avg_img = np.frombuffer(file.read(2*dimensions[0]*dimensions[1] + metadata_size[0] + metadata_size[1]), 
                                dtype=np.uint16)[metadata_size[0]//2:-metadata_size[1]//2].astype(np.float64).reshape(dimensions)
        num_imgs = 1
        
        # iterate through the rest of the files.
        # We want to avoid an overflow, so for each file we want to multiply by the old number of images
        # and divide by the new number to maintain this as the average
        continue_reading = True
        while continue_reading:
            try:
                new_img = np.frombuffer(file.read(2*dimensions[0]*dimensions[1] + metadata_size[0] + metadata_size[1]), 
                                dtype=np.uint16)[metadata_size[0]//2:-metadata_size[1]//2].astype(np.float64).reshape(dimensions)
                avg_img = avg_img*(num_imgs/(num_imgs + 1)) + new_img/(num_imgs + 1)
                num_imgs += 1
            except:
                continue_reading = False
                
        return avg_img
    
def img_stack(filename, dimensions=(400, 640), metadata_size=(5120, 5120)):
    """
    Get the entire stack of images as a numpy ndarray
    """
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as file:
        # get metadata size
        file.seek(-8, 2)
        metadata_json_size = int.from_bytes(file.read(8), byteorder='little')
        data_size = file_size - metadata_json_size - 9 # metadata is followed by an extra '\0' byte and then 8 bytes of size count
        num_frames = data_size//int(2*dimensions[0]*dimensions[1] + metadata_size[0] + metadata_size[1])
        full_stack = np.empty((num_frames, dimensions[0], dimensions[1]))
        file.seek(0)
        for i in range(num_frames):
            full_stack[i,:,:] = np.frombuffer(file.read(2*dimensions[0]*dimensions[1] + metadata_size[0] + metadata_size[1]), 
                                dtype=np.uint16)[metadata_size[0]//2:-metadata_size[1]//2].reshape(dimensions)
        return full_stack
    