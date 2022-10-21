import numpy as np
import scipy.io
def pad2d (x):
    Ny=np.shape(x)[0]
    Nx=np.shape(x)[1]
    return np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2)),'constant', constant_values=(0))

def pad2d_weights (x):
    Ny=np.shape(x)[0]
    Nx=np.shape(x)[1]
    return np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2), (0,0)),'constant', constant_values=(0))

def pad4d(x):
    Ny=np.shape(x)[0]
    Nx=np.shape(x)[1]
    return np.pad(x,((Ny//2,Ny//2),(Nx//2,Nx//2),(0,0),(0,0)),'constant', constant_values=(0))

def crop4d(x,rcL,rcU,ccL,ccU):
    return x[rcL:rcU,ccL:ccU,:,:]

def crop2d(x,rcL,rcU,ccL,ccU):   
    return x[rcL:rcU,ccL:ccU]

def nocrop(x):
    return x

def nopad(x):
    return x

def A_2d_svd(x,H,weights,pad,mode='shift_variant', extra_shift=True): #NOTE, H is already padded outside to save memory
    x=pad(x)
    Y=np.zeros((np.shape(x)[0],np.shape(x)[1]))
        
    if (mode =='shift_variant'):
        for r in range (0,np.shape(weights)[2]):
            #X=np.fft.fft2((np.multiply(pad(weights[:,:,r]),x)))
            X=np.fft.fft2(np.multiply(pad(weights[:,:,r]),x), axes=(0,1))
            Y=Y+ np.multiply(X,H[:,:,r])
    if extra_shift:
        return np.real((np.fft.ifftshift(np.fft.ifft2(Y))))
    return np.real(np.fft.ifft2(Y, axes=(0,1)))

def my_A_2d_svd(x,H,weights,pad,mode='shift_variant', extra_shift=True): #NOTE, H is already padded outside to save memory
    '''
    My version of the above function, which I think will be clearer and do a better job.
    '''
    x=pad(x)
    wx = x.reshape(x.shape[0], x.shape[1], 1)*pad2d_weights(weights)
    WX = np.fft.fft2(wx, axes=(0,1)) # shape is (Nx, Ny, K) where we did a rank-K SVD. We are doing FFT in the first two axes,
    # leaving the r-axis (which signifies index of eigen-PSF) untouched.
    return np.real( # cast to real. Result IS real to begin with, but is of complex type because IFFT returns complex in general
        #- # we want to invert the final image, because it will be returned inverted (not sure why...)
        np.sum( # sum over the K largest singular values. In my notebook this is summation through r
            np.fft.ifftshift(np.fft.ifft2( # inverse-fourier transform, breaks without the ifftshift
                WX*H # multiplication in the Fourier domain is a convolution in the original domain
            ,axes=(0,1))) # again, leave r-axis untouched
        ,-1) # this bracket closes summation through the r-axis, which is the last axis (hence -1).
    )
    
    

def A_2d(x,psf,pad):
    X=np.fft.fft2((pad(x)))
    H=np.fft.fft2((pad(psf)))
    Y=np.multiply(X,H)
    
    return np.real((np.fft.ifftshift(np.fft.ifft2(Y))))

def A_2d_adj_svd(Hconj,weights,y,pad):
    y=pad(y)
    x=np.zeros((np.shape(y)[0],np.shape(y)[1]))
    for r in range (0, np.shape(weights)[2]):
        x=x+np.multiply(pad(weights[:,:,r]),(np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(Hconj[:,:,r], np.fft.fft2((y))))))))
    #note the weights are real so we dont take the complex conjugate of it, which is the adjoint of the diag 
    return x

def A_2d_adj(y,psf,pad):
    H=np.fft.fft2((pad(psf)))
    Hconj=np.conj(H)
    x=(np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(Hconj, np.fft.fft2((pad(y))))))))
    
    return x

def A_3d(x,h,pad):
    #h is the psf stack
    #x is the variable to convolve with h
    x=pad(x)
    B=np.zeros((np.shape(x)[0],np.shape(x)[1]))
        

    for z in range (0,np.shape(h)[2]):
        #X=np.fft.fft2((np.multiply(pad(weights[:,:,z]),x)))
        B=B+ np.multiply(np.fft.fft2(x[:,:,z]),np.fft.fft2(pad(h[:,:,z])))
    
    return np.real((np.fft.ifftshift(np.fft.ifft2(B))))


def A_3d_svd(v,alpha,H,pad):
    #alpha is Ny-Nx-Nz-Nr, weights
    #v is Ny-Nx-Nz
    #H is Ny-Nx-Nz-Nr
    # b= sum_r (sum_z (h**alpra.*v))
    b=np.zeros((np.shape(v)[0],np.shape(v)[1]))
    for r in range (np.shape(H)[3]):
        for z in range (np.shape(H)[2]):
            b=b+np.multiply(H[:,:,z,r],np.fft.fft2(np.multiply(v[:,:,z],alpha[:,:,z,r])))
    
    return np.real(np.fft.ifftshift(np.fft.ifft2(b)))

def A_3d_adj(x,h,pad):
    y=np.zeros(np.shape(h))
    X=np.fft.fft2(pad(x))
    for z in range(np.shape(h)[2]):
        H=np.conj(np.fft.fft2(pad(h[:,:,z])))
        y[:,:,z]=np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(H,X))))
    return y

def A_3d_adj_svd(b,alpha,Hconj,pad):
    #y=sum_r(alpha.*H_conj**b)
    y=np.zeros((np.shape(alpha)[0],np.shape(alpha)[1],np.shape(alpha)[2]))
    B=np.fft.fft2(pad(b))
    for z in range(np.shape(alpha)[2]):
        for r in range(np.shape(alpha)[3]):
            y[:,:,z]=y[:,:,z]+np.multiply(alpha[:,:,z,r],np.fft.ifftshift(np.fft.ifft2(np.multiply(B,Hconj[:,:,z,r]))))
        
    return y

def grad(v):
    return np.array(np.gradient(v))  #returns gradient in x and in y


def grad_adj(v):  #adj of gradient is negative divergence
    z = np.zeros((n,n)) + 1j
    z -= np.gradient(v[0,:,:])[0]
    z -= np.gradient(v[1,:,:])[1]
    return z

def sim_data(im,H,weights,crop_indices, add_noise=True, extra_shift=True, a_svd_func=A_2d_svd):
    # ADDED BY ME
    # replace magic numbers with the actual dimensions of the image
    width = im.shape[1]
    height = im.shape[0]
    
    mu=0
    sigma=np.random.rand(1)*0.02+0.005 #abit much maybe 0.04 best0.04+0.01
    PEAK=np.random.rand(1)*1000+50

    I=im/np.max(im)
    #I[I<0.12]=0
    #sim=crop2d(A_2d_svd(I,H,weights,pad2d,extra_shift=extra_shift),*crop_indices)
    sim=crop2d(a_svd_func(I,H,weights,pad2d,extra_shift=extra_shift),*crop_indices)
    sim=sim/np.max(sim)
    sim=np.maximum(sim,0)

    p_noise = np.random.poisson(sim * PEAK)/PEAK

    #g_noise= np.random.normal(mu, sigma, 648*486)
    #g_noise=np.reshape(g_noise,(486,648))
    if add_noise:
        g_noise= np.random.normal(mu, sigma, width*height)
        g_noise=np.reshape(g_noise,(height,width))
        sim=sim+g_noise+p_noise
    sim=sim/np.max(sim)
    sim=np.maximum(sim,0)
    sim=sim/np.max(sim)
    return sim


# load in forward model weights
def load_weights(h_path='/home/kyrollos/LearnedMiniscope3D/RandoscopePSFS/SVD_2_5um_PSF_5um_1_ds4_dsz1_comps_green_SubAvg.mat',
                weights_path = '/home/kyrollos/LearnedMiniscope3D/RandoscopePSFS/SVD_2_5um_PSF_5um_1_ds4_dsz1_weights_green_SubAvg.mat'):
    h=scipy.io.loadmat(h_path) 
    weights=scipy.io.loadmat(weights_path )

    depth_plane=0 #NOTE Z here is 1 less than matlab file as python zero index. So this is z31 in matlab

    h=h['array_out']
    weights=weights['array_out']
    # make sure its (x,y,z,r)
    h=np.swapaxes(h,2,3)
    weights=np.swapaxes(weights,2,3)

    h=h[:,:,depth_plane,:]
    weights=weights[:,:,depth_plane,:]

    # Normalize weights to have maximum sum through rank of 1
    weights_norm = np.max(np.sum(weights[np.shape(weights)[0]//2-1,np.shape(weights)[1]//2-1,:],0))
    weights = weights/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=h/np.linalg.norm(np.ravel(h))

    # padded values for 2D

    ccL = np.shape(h)[1]//2
    ccU = 3*np.shape(h)[1]//2
    rcL = np.shape(h)[0]//2
    rcU = 3*np.shape(h)[0]//2

    H=np.ndarray((np.shape(h)[0]*2,np.shape(h)[1]*2,np.shape(h)[2]), dtype=complex)
    Hconj=np.ndarray((np.shape(h)[0]*2,np.shape(h)[1]*2,np.shape(h)[2]),dtype=complex)
    for i in range (np.shape(h)[2]):
        H[:,:,i]=(np.fft.fft2(pad2d(h[:,:,i])))
        Hconj[:,:,i]=(np.conj(H[:,:,i]))
    return H,weights,[rcL,rcU,ccL,ccU]

# load in forward model weights
def load_weights_2d(h_path='../data/nV3_h.mat',
                weights_path = '../data/nV3_weights.mat'):
    h=scipy.io.loadmat(h_path) 
    weights=scipy.io.loadmat(weights_path )

    #depth_plane=0 #NOTE Z here is 1 less than matlab file as python zero index. So this is z31 in matlab

    h=h['array_out']
    weights=weights['array_out']
    # make sure its (x,y,z,r)
    #h=np.swapaxes(h,2,3)
    #weights=np.swapaxes(weights,2,3)

    #h=h[:,:,depth_plane,:]
    #weights=weights[:,:,depth_plane,:]

    # Normalize weights to have maximum sum through rank of 1
#     weights_norm = np.max(np.sum(weights[np.shape(weights)[0]//2-1,np.shape(weights)[1]//2-1,:],0))
    weights_norm = np.absolute(np.max(np.sum(weights[np.shape(weights)[0]//2-1,np.shape(weights)[1]//2-1,:],0)))
    weights = weights/weights_norm;

    #normalize by norm of all stack. Can also try normalizing by max of all stack or by norm of each slice
    h=h/np.linalg.norm(np.ravel(h))

    # padded values for 2D

    ccL = np.shape(h)[1]//2
    ccU = 3*np.shape(h)[1]//2
    rcL = np.shape(h)[0]//2
    rcU = 3*np.shape(h)[0]//2

    H=np.ndarray((np.shape(h)[0]*2,np.shape(h)[1]*2,np.shape(h)[2]), dtype=complex)
    Hconj=np.ndarray((np.shape(h)[0]*2,np.shape(h)[1]*2,np.shape(h)[2]),dtype=complex)
    for i in range (np.shape(h)[2]):
        H[:,:,i]=(np.fft.fft2(pad2d(h[:,:,i])))
        Hconj[:,:,i]=(np.conj(H[:,:,i]))
    return H,weights,[rcL,rcU,ccL,ccU]