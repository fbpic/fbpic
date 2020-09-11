# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the SpectralTransformer object, 
by initializing a random array and testing the result of
the transform with the gpu and cpu version.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_cuda_transform.py
"""
import numpy as np
from fbpic.fields.spectral_transform import SpectralTransformer
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from numba import cuda
    from fbpic.utils.cuda import cupy
import time

if __name__ == '__main__' :

    # Parameters
    Nz = 2048
    Nr = 256
    rmax = 50.e-6
    m = 0

    # Initialize the random test_field
    interp_field_r = np.random.rand(Nz, Nr) + 1.j*np.random.rand(Nz, Nr)
    interp_field_t = np.random.rand(Nz, Nr) + 1.j*np.random.rand(Nz, Nr)
    d_interp_field_r = cupy.asarray( interp_field_r )
    d_interp_field_t = cupy.asarray( interp_field_t )
    # Initialize the field in spectral space
    spect_field_p = np.empty_like( interp_field_r )
    spect_field_m = np.empty_like( interp_field_t )
    d_spect_field_p = cupy.asarray( spect_field_p )
    d_spect_field_m = cupy.asarray( spect_field_m )
    # Initialize the field after back and forth transformation
    back_field_r = np.empty_like( interp_field_r )
    back_field_t = np.empty_like( interp_field_t )
    d_back_field_r = cupy.asarray( back_field_r )
    d_back_field_t = cupy.asarray( back_field_t )

    # ----------------
    # Scalar transform
    # ----------------
    print( '\n ### Scalar transform \n' )
    
    # Perform the transform on the CPU
    trans_cpu = SpectralTransformer( Nz, Nr, m, rmax )
    # Do a loop so as to get the fastest time
    # and remove compilation time
    tmin = 1.
    for i in range(10) :
        s = time.time()
        trans_cpu.interp2spect_scal( interp_field_r, spect_field_p )
        trans_cpu.spect2interp_scal( spect_field_p, back_field_r )
        e = time.time()
        tmin = min(tmin, e-s )
    print( '\n Time taken on the CPU : %.3f ms\n' %(tmin*1e3) )
    
    # Perform the transform on the GPU
    trans_gpu = SpectralTransformer( Nz, Nr, m, rmax, use_cuda=True )
    # Do a loop so as to get the fastest time
    # and remove compilation time
    tmin = 1.
    for i in range(10) :
        s = time.time()
        trans_gpu.interp2spect_scal( d_interp_field_r, d_spect_field_p )
        trans_gpu.spect2interp_scal( d_spect_field_p, d_back_field_r )
        e = time.time()
        tmin = min(tmin, e-s )
    print( '\n Time taken on the GPU : %.3f ms\n' %(tmin*1e3) )
    
    # Check accuracy
    check_spect_field_p = d_spect_field_p.get()
    check_back_field_r = d_back_field_r.get()
    print( 'Max error on forward transform : %e' \
      % abs(spect_field_p - check_spect_field_p).max() )
    print( 'Max error on backward transform : %e\n' \
      % abs(back_field_r - check_back_field_r).max() )

    # ----------------
    # Vector transform
    # ----------------
    print( '\n ### Vector transform \n' )
    
    # Perform the transform on the CPU
    trans_cpu = SpectralTransformer( Nz, Nr, m, rmax )
    # Do a loop so as to get the fastest time
    # and remove compilation time
    tmin = 1.
    for i in range(10) :
        s = time.time()
        trans_cpu.interp2spect_vect( interp_field_r, interp_field_t,
                                     spect_field_p, spect_field_m )
        trans_cpu.spect2interp_vect( spect_field_p, spect_field_m,
                                     back_field_r, back_field_t )
        cuda.synchronize()
        e = time.time()
        tmin = min(tmin, e-s )
    print( '\n Time taken on the CPU : %.3f ms\n' %(tmin*1e3) )
    
    # Perform the transform on the GPU
    trans_gpu = SpectralTransformer( Nz, Nr, m, rmax, use_cuda=True )
    # Do a loop so as to get the fastest time
    # and remove compilation time
    tmin = 1.
    for i in range(10) :
        s = time.time()
        trans_gpu.interp2spect_vect( d_interp_field_r, d_interp_field_t,
                                     d_spect_field_p, d_spect_field_m )
        trans_gpu.spect2interp_vect( d_spect_field_p, d_spect_field_m,
                                     d_back_field_r, d_back_field_t )
        cuda.synchronize()
        e = time.time()
        tmin = min(tmin, e-s )
    print( '\n Time taken on the GPU : %.3f ms\n' %(tmin*1e3) )
    
    # Check accuracy
    check_spect_field_p = d_spect_field_p.get()
    check_spect_field_m = d_spect_field_m.get()
    check_back_field_r = d_back_field_r.get()
    check_back_field_t = d_back_field_t.get()
    print( 'Max error on forward transform : %e' \
      % ( abs(spect_field_p - check_spect_field_p).max() \
      + abs(spect_field_m - check_spect_field_m).max() ) )
    print( 'Max error on backward transform : %e\n' \
      % ( abs(back_field_r - check_back_field_r).max() \
      + abs(back_field_t - check_back_field_t).max() ) )

      
