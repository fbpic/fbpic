# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FBPIC)
It defines the class that performs calculation of Monte-Carlo collisions.
"""
import numpy as np
import time
import math as m
import cupy
from scipy.constants import c, k, epsilon_0

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed, cupy_installed
from fbpic.utils.printing import catch_gpu_memory_error
from ..cuda_numba_utils import allocate_empty

if cupy_installed:
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from numba.cuda.random import create_xoroshiro128p_states
    from .cuda_methods import perform_collisions_cuda, \
        density_per_cell_cuda, n12_per_cell_cuda, \
        temperature_per_cell_cuda, pairs_per_cell_cuda, \
        get_cell_idx_per_pair

class MCCollisions(object):
    """
    Simulate Monte-Carlo (MC) collisions. The calculations randomly
    pair all particles in a cell and perform collisions
    """
    def __init__( self, species1, species2, use_cuda=False,
                coulomb_log = 0. ):
        """
        Initialize Monte-Carlo collisions

        Parameters
        ----------
        species1 : an fbpic Particles object

        species2 : an fbpic Particles object

        use_cuda : bool
        Whether the simulation is set up to use CUDA

        coulomb_log: float, optional
        """
        self.species1 = species1
        self.species2 = species2

        self.coulomb_log = coulomb_log
        
        # Particle pairs are processed in batches of size `batch_size`
        self.batch_size = 16

        self.use_cuda = use_cuda

    def handle_collisions( self, fld, dt ):
        """
        Handle collisions

        Parameters:
        -----------
        fld :`Fields` object which contains the field information

        dt : The simulation timestep
        """
        if self.use_cuda:
            self.handle_collisions_gpu( fld, dt )

    @catch_gpu_memory_error
    def handle_collisions_gpu( self, fld, dt ):
            """
            Handle collisions on GPU

            Parameters:
            -----------
            fld :`Fields` object which contains the field information

            dt : The simulation timestep
            """
            if self.species1.sorted == False:
                self.species1.sort_particles(fld = fld)
                self.species1.sorted = True
            if self.species2.sorted == False:
                self.species2.sort_particles(fld = fld)
                self.species2.sorted = True

            assert self.species1.sorted == True
            assert self.species2.sorted == True

            # Short-cut for use_cuda
            use_cuda = self.use_cuda

            prefix_sum1 = self.species1.prefix_sum
            prefix_sum2 = self.species2.prefix_sum

            Nz = fld.Nz

            d_invvol = fld.interp[0].d_invvol

            ux1 = getattr( self.species1, 'ux')
            uy1 = getattr( self.species1, 'uy')
            uz1 = getattr( self.species1, 'uz')
            w1 = getattr( self.species1, 'w')

            ux2 = getattr( self.species2, 'ux')
            uy2 = getattr( self.species2, 'uy')
            uz2 = getattr( self.species2, 'uz')
            w2 = getattr( self.species2, 'w')

            m1 = self.species1.m
            m2 = self.species2.m
            q1 = self.species1.q
            q2 = self.species2.q

            assert cupy.any(cupy.isnan(ux1)) == False
            assert cupy.any(cupy.isnan(uy1)) == False
            assert cupy.any(cupy.isnan(uz1)) == False
            assert cupy.any(cupy.isnan(ux2)) == False
            assert cupy.any(cupy.isnan(uy2)) == False
            assert cupy.any(cupy.isnan(uz2)) == False
            assert cupy.any(cupy.isnan(w1)) == False
            assert cupy.any(cupy.isnan(w2)) == False

            N_cells = int( prefix_sum1.shape[0] )
            npart1 = allocate_empty(N_cells, use_cuda, dtype=np.int32)
            npart2 = allocate_empty(N_cells, use_cuda, dtype=np.int32)

            # Calculate number of particles in cell
            npart1[0] = prefix_sum1[0]
            npart1[1:] = prefix_sum1[1:] - prefix_sum1[:-1]
            npart2[0] = prefix_sum2[0]
            npart2[1:] = prefix_sum2[1:] - prefix_sum2[:-1]

            assert cupy.any(cupy.isnan(npart1)) == False
            assert cupy.any(cupy.isnan(npart2)) == False

            assert cupy.sum( npart1 ) > 0
            assert cupy.sum( npart2 ) > 0

            # Calculate number of pairs in cell
            npairs = allocate_empty(N_cells, use_cuda, dtype=np.int32)
            bpg, tpg = cuda_tpb_bpg_1d( N_cells )
            pairs_per_cell_cuda[ bpg, tpg ](N_cells, npart1, npart2, npairs)

            # Calculate density of species 1 in each cell
            n1 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            density_per_cell_cuda[ bpg, tpg ]( N_cells, n1, w1,
                                npart1, prefix_sum1,
                                d_invvol, Nz )
            
            # Calculate density of species 2 in each cell
            n2 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            density_per_cell_cuda[ bpg, tpg ]( N_cells, n2, w2,
                                npart2, prefix_sum2,
                                d_invvol, Nz )
            
            # Calculate sum of minimum weights in each cell
            n12 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            n12_per_cell_cuda[ bpg, tpg ]( N_cells, n12, w1, w2, npart1,
                                            prefix_sum1, prefix_sum2 )

            # Calculate temperature in each cell
            temperature1 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            temperature_per_cell_cuda[ bpg, tpg ]( N_cells, temperature1, npart1,
                                prefix_sum1, m1,
                                ux1, uy1, uz1 )

            temperature2 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            temperature_per_cell_cuda[ bpg, tpg ]( N_cells, temperature2, npart2,
                                prefix_sum2, m2,
                                ux2, uy2, uz2 )

            mean_T1 = cupy.sum( temperature1 ) / N_cells
            mean_T2 = cupy.sum( temperature2 ) / N_cells
            mean_n1 = cupy.sum( n1 ) / N_cells
            mean_n2 = cupy.sum( n2 ) / N_cells
            mean_n12 = cupy.sum( n12 ) / N_cells

            print("\n mean_T1 = ", mean_T1)
            print("mean_T2 = ", mean_T2)
            print("mean_n1 = ", mean_n1)
            print("mean_n2 = ", mean_n2)
            print("mean_n12 = ", mean_n12)

            # Total number of pairs
            npairs_tot = int( cupy.sum( npairs ) )
            print("\n Total number of pairs = ", npairs_tot)

            # Cumulative prefix sum of pairs
            prefix_sum_pair = cupy.cumsum(npairs)

            # Cell index of pair
            cell_idx = cupy.empty(npairs_tot, dtype=np.int64)
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( N_cells )
            get_cell_idx_per_pair[dim_grid_1d, dim_block_1d](N_cells, cell_idx, npairs, prefix_sum_pair)

            # The particles need to be shuffled, in each cell, in non-repeating order
            # so that particles are randomly paired once
            
            # Process particle pairs in batches
            N_batch = int( npairs_tot  / self.batch_size ) + 1
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_batch, seed )

            bpg, tpg = cuda_tpb_bpg_1d( N_batch )
            perform_collisions_cuda[ bpg, tpg ]( prefix_sum1, prefix_sum2,
                        cell_idx, npart1, npart2, npairs_tot,
                        n1, n2, n12, m1, m2,
                        q1, q2, w1, w2,
                        ux1, uy1, uz1,
                        ux2, uy2, uz2,
                        dt, self.coulomb_log,
                        temperature1, temperature2,
                        random_states )

            if cupy.any(cupy.isnan(ux1)) == True or \
                cupy.any(cupy.isnan(uy1)) == True or \
                cupy.any(cupy.isnan(uz1)) == True or \
                cupy.any(cupy.isnan(ux2)) == True or \
                cupy.any(cupy.isnan(uy2)) == True or \
                cupy.any(cupy.isnan(uz2)) == True:
                print("\n NaN solution.")
                return
            else:
                setattr(self.species1.ux, 'ux', ux1)
                setattr(self.species1.uy, 'uy', uy1)
                setattr(self.species1.uz, 'uz', uz1)
                setattr(self.species2.ux, 'ux', ux2)
                setattr(self.species2.uy, 'uy', uy2)
                setattr(self.species2.uz, 'uz', uz2)
