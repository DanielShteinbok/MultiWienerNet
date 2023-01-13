import tensorflow as tf
from tensorflow.keras import layers
from models.layers import *

def conv2d_block(x, filters, kernel_size, padding='same', dilation_rate=1, batch_norm=True, activation='relu'):
    """
    Applies Conv2D - BN - ReLU block.
    """
    x = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=False)(x)
    
    if batch_norm:
        x = layers.BatchNormalization()(x)

    if activation is not None:
        x = layers.Activation(activation)(x)
    
    return x


def encoder_block(x, filters, kernel_size, padding='same', dilation_rate=1, pooling='max'):
    """
    Encoder block used in contracting path of UNet.
    """
    # EXAMINE: line below is repeated twice? Is this a mistake?
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    x_skip = x
#     print(x.shape)
    if pooling == 'max':
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    elif pooling == 'average':
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    else:
        assert False, 'Pooling layer {} not implemented'.format(pooling)
    
    return x, x_skip


def decoder_block(x, x_skip, filters, kernel_size, padding='same', dilation_rate=1):
    """
    Decoder block used in expansive path of UNet.
    """
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    
    # Calculate cropping for down_tensor to concatenate with x
    
    if x_skip is not None:
        _, h2, w2, _ = x_skip.shape
        _, h1, w1, _ = x.shape
        h_diff, w_diff = h2 - h1, w2 - w1

        cropping = ((int(np.ceil(h_diff / 2)), int(np.floor(h_diff / 2))),
                    (int(np.ceil(w_diff / 2)), int(np.floor(w_diff / 2))))
        x_skip = layers.Cropping2D(cropping=cropping)(x_skip)
        x = layers.concatenate([x, x_skip], axis=3)

    # EXAMINE: line below is repeated THRICE? Is this a mistake???
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    
    return x


################################################################################################################################################


def decoder_block_resize(x, x_skip, filters, kernel_size, padding='same', dilation_rate=1):
    """
    Decoder block used in expansive path of UNet. Unlike before, this block resizes the skip connections rather than cropping.
    """
#     print(x.shape)
#     print(x_skip.shape[1:3])
    x = tf.image.resize(x, x_skip.shape[1:3], method='nearest')
    
    x = layers.concatenate([x, x_skip], axis=3)
    # EXAMINE: line below is repeated THRICE? Is this a mistake???
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    x = conv2d_block(x, filters, kernel_size, padding, dilation_rate, batch_norm=True, activation='relu')
    
    return x


def UNet(height, width, encoding_cs=[24, 64, 128, 256, 512, 1024], 
         center_cs=1024,
         decoding_cs=[512, 256, 128, 64, 24, 24],
         skip_connections=[True, True, True, True, True, False]):
    
    """
    Basic UNet which does not require cropping.
    
    Inputs:
        - height: input height
        - width: input width
        - encoding_cs: list of channels along contracting path
        - decoding_cs: list of channels along expansive path
    """

    inputs = tf.keras.Input((height, width, 1))
    
    x = inputs
    
    skips = []
    
    # Contracting path
    for c in encoding_cs:
        x, x_skip = encoder_block(x, c, kernel_size=3, padding='same', dilation_rate=1, pooling='average')
        skips.append(x_skip)

    skips = list(reversed(skips))
    
    # Center
    x = conv2d_block(x, center_cs, kernel_size=3, padding='same')
    
    # Expansive path
    for i, c in enumerate(decoding_cs):
        if skip_connections[i]:
            x = decoder_block_resize(x, skips[i], c, kernel_size=3, padding='same', dilation_rate=1)
        else:
            x = decoder_block(x, None, c, kernel_size=3, padding='same', dilation_rate=1)
        
    # Classify
    x = layers.Conv2D(filters=1, kernel_size=1, use_bias=True, activation='relu')(x)
#     outputs=x
    outputs = tf.squeeze(x, axis=3)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model



def UNet_multiwiener_resize(height, width, initial_psfs, initial_Ks, 
                     encoding_cs=[24, 64, 128, 256, 512, 1024], 
                     center_cs=1024,
                     decoding_cs=[512, 256, 128, 64, 24, 24], 
                     skip_connections=[True, True, True, True, True, True],
                     psfs_trainable=True,
                     Ks_trainable=True,
                     training_noise=False):
    """
    Multiwiener UNet which doesn't require cropping.
    
    Inputs:
        - height: input height
        - width: input width
        - initial_psfs: preinitialized psfs
        - initial_Ks: regularization terms for Wiener deconvolutions
        - encoding_cs: list of channels along contracting path
        - decoding_cs: list of channels along expansive path
        - skip_connections: list of boolean to determine whether to concatenate with decoding channel at that index
    """
    # ADDED:
    # initial_psfs.shape is (486, 648, 9), i.e. (H, W, C)
    # ensure the PSF is no larger than requirements, in which case throw an error
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
        

    inputs = tf.keras.Input((height, width, 1))
    
    x = inputs
    
    # ADDED BY DANIEL: add Gaussian noise to images that matches the actual nV3 images
    if training_noise:
        x = tf.keras.layers.GaussianNoise(4.2)(x)
    
    # Multi-Wiener deconvolutions
    x = MultiWienerDeconvolution(initial_psfs, initial_Ks, psfs_trainable=psfs_trainable, Ks_trainable=Ks_trainable)(x)
    
    skips = []
    
    # Contracting path
    for c in encoding_cs:
        x, x_skip = encoder_block(x, c, kernel_size=3, padding='same', dilation_rate=1, pooling='average')
        skips.append(x_skip)

    skips = list(reversed(skips))
    
    # Center
    x = conv2d_block(x, center_cs, kernel_size=3, padding='same')
    
    # Expansive path
    for i, c in enumerate(decoding_cs):
        if skip_connections[i]:
            x = decoder_block_resize(x, skips[i], c, kernel_size=3, padding='same', dilation_rate=1)
        else:
            x = decoder_block(x, None, c, kernel_size=3, padding='same', dilation_rate=1)
        
    # Classify
    x = layers.Conv2D(filters=1, kernel_size=1, use_bias=True, activation='relu')(x)
    outputs = tf.squeeze(x, axis=3)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model



def UNet_wiener(height, width, initial_psf, initial_K, 
                     encoding_cs=[24, 64, 128, 256, 512, 1024], 
                     center_cs=1024,
                     decoding_cs=[512, 256, 128, 64, 24, 24], 
                     skip_connections=[True, True, True, True, True, True]):
    """
    Single Wiener UNet which doesn't require cropping.
    
    Inputs:
        - height: input height
        - width: input width
        - initial_psf: preinitialized psf
        - initial_K: regularization term for Wiener deconvolution
        - encoding_cs: list of channels along contracting path
        - decoding_cs: list of channels along expansive path
        - skip_connections: list of boolean to determine whether to concatenate with decoding channel at that index
    """

    inputs = tf.keras.Input((height, width, 1))
    
    x = inputs
    
    # Multi-Wiener deconvolutions
    x = WienerDeconvolution(initial_psf, initial_K)(x)
    
    skips = []
    
    # Contracting path
    for c in encoding_cs:
        x, x_skip = encoder_block(x, c, kernel_size=3, padding='same', dilation_rate=1, pooling='average')
        skips.append(x_skip)

    skips = list(reversed(skips))
    
    # Center
    x = conv2d_block(x, center_cs, kernel_size=3, padding='same')
    
    # Expansive path
    for i, c in enumerate(decoding_cs):
        if skip_connections[i]:
            x = decoder_block_resize(x, skips[i], c, kernel_size=3, padding='same', dilation_rate=1)
        else:
            x = decoder_block(x, None, c, kernel_size=3, padding='same', dilation_rate=1)
        
    # Classify
    x = layers.Conv2D(filters=1, kernel_size=1, use_bias=True, activation='relu')(x)
    outputs = tf.squeeze(x, axis=3)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model




