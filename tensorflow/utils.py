import numpy as np
import tensorflow as tf
import math

def normalize(x):
    """
    Normalizes numpy array to [0, 1].
    """
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)

def pad_2d(x, mode='constant'):
    """
    Pads 2d array x before FFT convolution.
    """
    _, _, h, w = x.shape
    padding = ((0, 0), (0, 0), 
               (int(np.ceil(h / 2)), int(np.floor(h / 2))),
               (int(np.ceil(w / 2)), int(np.floor(w / 2))))
    x = np.pad(x, pad_width=padding, mode=mode)
    return x

def pad_2d_tf(x, mode='CONSTANT', axes=(-2, -1)):
    """
    Fix pad_2d to allow variable length dimensions. 
    """
    n_dim = len(x.shape)
    axes = np.array(axes) % n_dim

    h, w = np.array(x.shape)[axes]
    padding = np.array([(0, 0)] * n_dim)
    padding[axes] = (int(np.ceil(h / 2)), int(np.floor(h / 2))), (int(np.ceil(w / 2)), int(np.floor(w / 2)))

    x = tf.pad(x, paddings=padding, mode=mode)
    return x



def crop_2d(v):
    """
    Crops 2d array x after FFT convolution. Inverse of pad2d.
    """
    h, w = v.shape
    h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
    w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
    return v[h1:h2, w1:w2]

def crop_2d_tf(v):
    """
    Crops 2d array v after FFT convolution. Inverse of pad2d.
    """
    n_dim = len(v.shape)
    
    if n_dim == 2:    
        h, w = v.shape
        h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
        w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
        return v[h1:h2, w1:w2]
    elif n_dim == 3:
        _, h, w = v.shape
        h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
        w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
        return v[:, h1:h2, w1:w2]
    elif n_dim == 4:
        _, h, w, _ = v.shape
        h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
        w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
        return v[:, h1:h2, w1:w2, :]
    
    elif n_dim == 5:
        _, h, w, _,_ = v.shape
        h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
        w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
        return v[:, h1:h2, w1:w2, :,:]
    
    
def calc_psnr(Iin,Itarget):
    
    mse=np.mean(np.square(Iin-Itarget))
    return 10*math.log10(1/mse)


def parse_function(inputname, outputname):

    # Read an image from a file
    input_string = tf.io.read_file(inputname)
    # Decode it into a dense vector
    input_decoded = tf.cast(tf.image.decode_png(input_string, channels=1), tf.float32)
    # Resize it to fixed shape
#     input_resized = tf.image.resize(input_decoded, [img_height, img_width])
    input_normalized = input_decoded / 255.0
    
    # Read an image from a file
    output_string = tf.io.read_file(outputname)
    # Decode it into a dense vector
    output_decoded =  tf.cast(tf.image.decode_png(output_string, channels=1), tf.float32)
    # Resize it to fixed shape
    # Normalize it from [0, 255] to [0.0, 1.0]
    output_normalized = output_decoded / 255.0
    
    return input_normalized, output_normalized


def configure_for_performance(ds,batch_size):  #shuffte, batch, and have batches available asap
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    return ds



def SSIMLoss(y_true, y_pred):
    y_true = y_true[..., np.newaxis]
    y_pred = y_pred[..., np.newaxis]
    
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def SSIMLoss_l1(y_true, y_pred):
    y_true = y_true[..., np.newaxis]
    y_pred = y_pred[..., np.newaxis]
    L1=tf.reduce_mean(tf.abs(y_true-y_pred))
    
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))+L1



def grad(model, myloss,inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = myloss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def grad_universal(model, myloss,inputs, targets, *args, **kwargs):
    """
    added by Daniel. 
    This is meant to be a more "universal" version of grad, 
    which takes a list and dictionary of additional arguments to pass to myloss.
    myloss must have the following signature:
    
    myloss(model, *args, **kwargs)
    """
    with tf.GradientTape() as tape:
        loss_value = myloss(model, inputs, targets, *args, **kwargs)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def SSIMLoss(y_true, y_pred):
    y_true = y_true[..., np.newaxis]
    y_pred = y_pred[..., np.newaxis]
    
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def SSIMLoss_l1(model,x,y_true):
    y_pred=model(x)
    y_pred = tf.expand_dims(y_pred, -1)
    loss_l1 = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return loss_l1+loss_ssim

def SSIMLoss_l1_indicator(model,x,y_true,indicator, training=True):
    """
    same as SSIMLoss_l1(model, x, y_true), 
    except here we take an indicator matrix and only evaluate within that indicator.
    
    indicator: a matrix broadcastable to model(x), e.g. with shape (800, 1280, 1)
    """
#     y_pred=model(x)
    y_pred=model(x, training=training)
    y_pred = tf.expand_dims(y_pred, -1)
    loss_l1 = tf.reduce_mean(tf.abs(y_pred - y_true)*indicator, axis=-1)
    loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(y_true*indicator, y_pred*indicator, 1.0))
    return loss_l1+loss_ssim

def SSIMLoss_l1_generator(training=True):
    """
    return a callback function that takes:
    callback(y_true, y_pred)
    """
#     y_pred=model(x)
    def loss_func(y_true, y_pred):
#     y_pred=model(x, training=training)
        print("y_true.shape: ", str(y_true.shape))
        print("y_pred.shape: ", str(y_pred.shape))
#         y_pred = tf.expand_dims(y_pred, -1)
        loss_l1 = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
#         if y_true.shape[1] is None:
#             print("reshaping y_true")
#             y_true += np.zeros_like(y_pred.shape)
        loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
#         loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(tf.multiply(y_true, indicator), tf.multiply(y_pred, indicator), 1.0))
        return loss_l1+loss_ssim
    return loss_func

def SSIMLoss_l1_indicator_generator(indicator, training=True):
    """
    return a callback function that takes:
    callback(y_true, y_pred)
    """
#     y_pred=model(x)
    def loss_func(y_true, y_pred):
#     y_pred=model(x, training=training)
        print("y_true.shape: ", str(y_true.shape))
        print("y_pred.shape: ", str(y_pred.shape))
#         y_pred = tf.expand_dims(y_pred, -1)
        loss_l1 = tf.reduce_mean(tf.abs(y_pred - y_true)*indicator, axis=-1)
    
        # evil hack to allow the model to compile. At that point y_true has dimensions (None, None, None, None)
        # and so the compilation fails because SSIM fails given dimensions less than 11x11
        # there are working examples using ssim successfully without this hack, 
        # but I suspect they don't have the (None, None, None, None) problem because this is caused by the multiwiener layer
        if y_true.shape[1] is not None:
            loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(y_true*indicator, y_pred*indicator, 1.0))
        else:
            loss_ssim=0.0
            print("y_true.shape[1] is None")
#         loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(tf.multiply(y_true, indicator), tf.multiply(y_pred, indicator), 1.0))
        return loss_l1+loss_ssim
    return loss_func

def Loss_l1_indicator_generator(indicator, training=True):
    """
    return a callback function that takes:
    callback(y_true, y_pred)
    """
#     y_pred=model(x)
    def loss_func(y_true, y_pred):
#     y_pred=model(x, training=training)
        print("y_true.shape: ", str(y_true.shape))
        print("y_pred.shape: ", str(y_pred.shape))
#         y_pred = tf.expand_dims(y_pred, -1)
        loss_l1 = tf.reduce_mean(tf.abs(y_pred - y_true)*indicator, axis=-1)
#         if y_true.shape[1] is None:
#             print("reshaping y_true")
#             y_true += np.zeros_like(y_pred.shape)
#         loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(y_true*indicator, y_pred*indicator, 1.0))
#         loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(tf.multiply(y_true, indicator), tf.multiply(y_pred, indicator), 1.0))
        return loss_l1
    return loss_func

class SSIMLoss_l1_indicator_Class(tf.keras.losses.Loss):
    def __init__(self, indicator, name="SSIM_and_1_norm"):
        super().__init__(name=name)
        self.indicator = indicator
        
    def call(self, y_true, y_pred):
        y_pred = tf.expand_dims(y_pred, -1)
        loss_l1 = tf.reduce_mean(tf.abs(y_pred - y_true)*self.indicator, axis=-1)
        loss_ssim=1.0 - tf.reduce_mean(tf.image.ssim(y_true*self.indicator, y_pred*self.indicator, 1.0))
        return loss_l1+loss_ssim