"""
This file tests the SpectralTransformer object, 
by initializing a random array and testing the result of
the transform with the gpu and cpu version.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_cuda_transform.py
"""
import numpy as np
from fbpic.fields.fields import SpectralTransformer
from fbpic.cuda_utils import *
from numba import cuda
import time

if __name__ == '__main__' :

    # Parameters
    Nz = 2048
    Nr = 256
    rmax = 50.e-6
    m = 0

    # Initialize the random test_field
    interp_field = np.random.rand(Nz, Nr) + 1.j*np.random.rand(Nz, Nr)
    d_interp_field = cuda.to_device( interp_field )
    # Initialize the field in spectral space
    spect_field = np.empty_like( interp_field )
    d_spect_field = cuda.to_device( spect_field )
    # Initialize the field after back and forth transformation
    back_field = np.empty_like( interp_field )
    d_back_field = cuda.to_device( back_field )
    
    # Perform the transform on the CPU
    trans_cpu = SpectralTransformer( Nz, Nr, m, rmax )
    # Do a loop so as to get the fastest time
    # and remove compilation time
    tmin = 1.
    for i in range(10) :
        s = time.time()
        trans_cpu.interp2spect_scal( interp_field, spect_field )
        trans_cpu.spect2interp_scal( spect_field, back_field )
        e = time.time()
        tmin = min(tmin, e-s )
    print '\n Time taken on the CPU : %.3f ms\n' %(tmin*1e3)
    
    # Perform the transform on the GPU
    trans_gpu = SpectralTransformer( Nz, Nr, m, rmax, use_cuda=True )
    # Do a loop so as to get the fastest time
    # and remove compilation time
    tmin = 1.
    for i in range(10) :
        s = time.time()
        trans_gpu.interp2spect_scal( d_interp_field, d_spect_field )
        trans_gpu.spect2interp_scal( d_spect_field, d_back_field )
        e = time.time()
        tmin = min(tmin, e-s )
    print '\n Time taken on the GPU : %.3f ms\n' %(tmin*1e3)
    
    # Check accuracy
    d_spect_field = d_spect_field.copy_to_host()
    d_back_field = d_back_field.copy_to_host()
    print 'Max error on forward transform : %e' \
      % abs(spect_field - d_spect_field).max()
    print 'Max error on backward transform : %e\n' \
      % abs(back_field - d_back_field).max()
