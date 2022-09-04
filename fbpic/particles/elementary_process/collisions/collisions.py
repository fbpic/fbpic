# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FBPIC)
It defines the class that performs calculation of Monte-Carlo collisions.
"""
import numpy as np
import math as m
import cupy
from scipy.constants import c, k, epsilon_0, e

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cupy_installed
from fbpic.utils.printing import catch_gpu_memory_error
from ..cuda_numba_utils import allocate_empty

if cupy_installed:
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from numba.cuda.random import create_xoroshiro128p_states
    from .cuda_methods import perform_collisions_cuda, \
        density_per_cell_cuda, n12_per_cell_cuda, \
        temperature_per_cell_cuda, pairs_per_cell_cuda, \
        get_cell_idx_per_pair_cuda, get_shuffled_idx_per_particle_cuda

class MCCollisions(object):
    """
    Simulate Monte-Carlo (MC) collisions. The calculations randomly
    pair all particles in a cell and perform collisions
    """
    def __init__( self, species1, species2, use_cuda=False,
                coulomb_log = 0.,
                start = 0,
                period = 0,
                debug = False ):
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
        self.batch_size = 10

        self.use_cuda = use_cuda

        self.start = start
        self.period = period
        self.debug = debug

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

            Nz = fld.Nz
            Nr = fld.Nr

            d_invvol = fld.interp[0].d_invvol

            N_cells = int( self.species1.prefix_sum.shape[0] )
            npart1 = cupy.zeros(N_cells, dtype=np.int32)
            npart2 = cupy.zeros(N_cells, dtype=np.int32)

            #print("\nrefix_sum1 = ", self.species1.prefix_sum)

            #print("\nprefix_sum2 = ", self.species2.prefix_sum)

            # Calculate number of particles in cell
            npart1[0] = self.species1.prefix_sum[0]
            npart1[1:] = self.species1.prefix_sum[1:] - self.species1.prefix_sum[:-1]
            npart2[0] = self.species2.prefix_sum[0]
            npart2[1:] = self.species2.prefix_sum[1:] - self.species2.prefix_sum[:-1]

            assert self.species1.Ntot > 0
            assert self.species2.Ntot > 0
            
            intra = True if self.species1 == self.species2 else False

            #print("\npart1 = ", npart1)

            #print("\npart2 = ", npart2)

            # Calculate number of pairs in cell
            npairs = cupy.zeros(N_cells, dtype=np.int32)
            bpg, tpg = cuda_tpb_bpg_1d( N_cells )
            pairs_per_cell_cuda[ bpg, tpg ](N_cells, npart1, npart2, npairs, intra)

            #print("\nnpairs = ", npairs)

            if self.species1.calc == False:
                # Calculate density of species 1 in each cell
                density_per_cell_cuda[ bpg, tpg ](N_cells, self.species1.density, self.species1.w,
                                    npart1, self.species1.prefix_sum,
                                    d_invvol, Nz)

                # Calculate temperature of species 1 in each cell
                temperature_per_cell_cuda[ bpg, tpg ](N_cells, self.species1.temperature, npart1,
                                    self.species1.prefix_sum, self.species1.m,
                                    self.species1.ux, self.species1.uy, self.species1.uz)
                self.species1.calc = True

            if not intra:
                if self.species2.calc == False:
                    # Calculate density of species 2 in each cell
                    density_per_cell_cuda[ bpg, tpg ](N_cells, self.species2.density, self.species2.w,
                                        npart2, self.species2.prefix_sum,
                                        d_invvol, Nz)
                    
                    # Calculate temperature of species 2 in each cell
                    temperature_per_cell_cuda[ bpg, tpg ](N_cells, self.species2.temperature, npart2,
                                        self.species2.prefix_sum, self.species2.m,
                                        self.species2.ux, self.species2.uy, self.species2.uz)
                    self.species2.calc = True

            # Total number of pairs
            npairs_tot = int( cupy.sum( npairs ) )
            #print("npairs_tot:", npairs_tot)

            # Cumulative prefix sum of pairs
            prefix_sum_pair = cupy.cumsum(npairs)

            # Cell index of pair
            cell_idx = cupy.zeros(npairs_tot, dtype=np.int32)
            get_cell_idx_per_pair_cuda[ bpg, tpg ](N_cells, cell_idx, npairs, prefix_sum_pair)

            #print(cell_idx)
            
            # Shuffled index of particle 1
            shuffled_idx1 = cupy.zeros(npairs_tot, dtype=np.int32)
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_cells, seed )
            get_shuffled_idx_per_particle_cuda[ bpg, tpg ](N_cells, shuffled_idx1, npart1,
                                                            npairs, prefix_sum_pair,
                                                            random_states, intra, 1)

            # Shuffled index of particle 2
            shuffled_idx2 = cupy.zeros(npairs_tot, dtype=np.int32)
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_cells, seed )
            get_shuffled_idx_per_particle_cuda[ bpg, tpg ](N_cells, shuffled_idx2, npart2,
                                                            npairs, prefix_sum_pair,
                                                            random_states, intra, 2)

            #print("shuffled idx2 = ", shuffled_idx2)
            
            # Calculate sum of minimum weights in each cell
            n12 = cupy.zeros(N_cells, dtype=np.float64)
            n12_per_cell_cuda[ bpg, tpg ](N_cells, n12, self.species1.w, self.species2.w,
                      npairs, shuffled_idx1, shuffled_idx2,
					  prefix_sum_pair, self.species1.prefix_sum, self.species2.prefix_sum,
                      d_invvol, Nz)
            
            
            param_s = cupy.zeros(npairs_tot, dtype=np.float64)
            param_logL = cupy.zeros(npairs_tot, dtype=np.float64)
            
            # Process particle pairs in batches
            N_batch = int( npairs_tot  / self.batch_size ) + 1
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_batch, seed )
            bpg, tpg = cuda_tpb_bpg_1d( N_batch )
            perform_collisions_cuda[ bpg, tpg ](N_batch, 
                        self.batch_size, npairs_tot,
                        self.species1.prefix_sum, self.species2.prefix_sum,
                        shuffled_idx1, shuffled_idx2, cell_idx,
                        self.species1.density, self.species2.density,
                        self.species1.temperature, self.species2.temperature,
                        n12, self.species1.m, self.species2.m,
                        self.species1.q, self.species2.q, 
                        self.species1.w, self.species2.w,
                        self.species1.ux, self.species1.uy, self.species1.uz,
                        self.species2.ux, self.species2.uy, self.species2.uz,
                        dt, self.coulomb_log,
                        random_states, self.period, self.debug,
                        param_s, param_logL)
            
            if self.debug:
                Ncp_1 = np.count_nonzero(self.species1.temperature)
                Ncp_2 = np.count_nonzero(self.species2.temperature)
                mean_T1 = cupy.sum( self.species1.temperature ) / Ncp_1
                mean_T2 = cupy.sum( self.species2.temperature ) / Ncp_2
                max_T1 = cupy.max( self.species1.temperature )
                max_T2 = cupy.max( self.species2.temperature )
                min_T1 = cupy.min( self.species1.temperature )
                min_T2 = cupy.min( self.species2.temperature )

                print("\n<T1> = ", (k / e) * mean_T1, " eV")
                print("min(T1) = ", (k / e) * min_T1, " eV")
                print("max(T1) = ", (k / e) * max_T1, " eV")

                print("<T2> = ", (k / e) * mean_T2, " eV")
                print("min(T2) = ", (k / e) * min_T2, " eV")
                print("max(T2) = ", (k / e) * max_T2, " eV")

                print("<s> = ", cupy.sum( param_s ) / npairs_tot)
                print("min(s) = ", cupy.min( param_s ))
                print("max(s) = ", cupy.max( param_s ))

                print("<logL> = ", cupy.sum( param_logL ) / npairs_tot)
                print("min(logL) = ", cupy.min( param_logL ))
                print("max(logL) = ", cupy.max( param_logL ))

