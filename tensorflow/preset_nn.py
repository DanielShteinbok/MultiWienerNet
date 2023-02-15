# generate an uninitialized multiwienernet that matches what I've been training
# so that you can test and use it outside of training
# prior to this, the code for initializing the multiwienernet was substantial
# and replicated across different notebooks, including the training notebook
# this created the nasty possibility for discrepancies, which would interfere
# with loading weights

import numpy as np
import models.model_2d as mod

def instant_multiwienernet(psfs_shape=(800,1280,21), pooling='average'):
    psfs = np.empty(psfs_shape)
    Ks =np.empty((1,1,psfs.shape[2]))
    model =mod.UNet_multiwiener_resize(psfs_shape[0], psfs_shape[1], psfs, Ks, 
             encoding_cs=[24, 64, 128, 256, 512, 1024],
             center_cs=1024,
             decoding_cs=[512, 256, 128, 64, 24, 24],
             skip_connections=[True, True, True, True, True, False], psfs_trainable=True,
                training_noise=True,
              pooling=pooling)
    return model