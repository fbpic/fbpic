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

            N_cells = int( prefix_sum1.shape[0] )
            npart1 = allocate_empty(N_cells, use_cuda, dtype=np.int32)
            npart2 = allocate_empty(N_cells, use_cuda, dtype=np.int32)

            # Calculate number of particles in cell
            npart1[0] = prefix_sum1[0]
            npart1[1:] = prefix_sum1[1:] - prefix_sum1[:-1]
            npart2[0] = prefix_sum2[0]
            npart2[1:] = prefix_sum2[1:] - prefix_sum2[:-1]

            assert cupy.sum( npart1 ) > 0
            assert cupy.sum( npart2 ) > 0
            
            intra = True if self.species1 == self.species2 else False

            # Calculate number of pairs in cell
            npairs = allocate_empty(N_cells, use_cuda, dtype=np.int32)
            bpg, tpg = cuda_tpb_bpg_1d( N_cells )
            pairs_per_cell_cuda[ bpg, tpg ](N_cells, npart1, npart2, npairs, intra)

            if self.species1.calc == False:
                # Calculate density of species 1 in each cell
                density_per_cell_cuda[ bpg, tpg ](N_cells, self.species1.density, w1,
                                    npart1, prefix_sum1,
                                    d_invvol, Nz)

                # Calculate temperature of species 1 in each cell
                temperature_per_cell_cuda[ bpg, tpg ](N_cells, self.species1.temperature, npart1,
                                    prefix_sum1, m1,
                                    ux1, uy1, uz1)
                self.species1.calc = True

            if not intra:
                if self.species2.calc == False:
                    # Calculate density of species 2 in each cell
                    density_per_cell_cuda[ bpg, tpg ](N_cells, self.species2.density, w2,
                                        npart2, prefix_sum2,
                                        d_invvol, Nz)
                    
                    # Calculate temperature of species 2 in each cell
                    temperature_per_cell_cuda[ bpg, tpg ](N_cells, self.species2.temperature, npart2,
                                        prefix_sum2, m2,
                                        ux2, uy2, uz2)
                    self.species2.calc = True

            # Total number of pairs
            npairs_tot = int( cupy.sum( npairs ) )

            # Cumulative prefix sum of pairs
            prefix_sum_pair = cupy.cumsum(npairs)

            # Cell index of pair
            cell_idx = cupy.empty(npairs_tot, dtype=np.int32)
            get_cell_idx_per_pair_cuda[ bpg, tpg ](N_cells, cell_idx, npairs, prefix_sum_pair)

            # Shuffled index of particle 1
            shuffled_idx1 = cupy.empty(npairs_tot, dtype=np.int32)
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_cells, seed )
            get_shuffled_idx_per_particle_cuda[ bpg, tpg ](N_cells, shuffled_idx1, npart1,
                                                            npairs, prefix_sum_pair,
                                                            random_states, intra, 1)

            # Shuffled index of particle 2
            shuffled_idx2 = cupy.empty(npairs_tot, dtype=np.int32)
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_cells, seed )
            get_shuffled_idx_per_particle_cuda[ bpg, tpg ](N_cells, shuffled_idx2, npart2,
                                                            npairs, prefix_sum_pair,
                                                            random_states, intra, 2)

            # Calculate sum of minimum weights in each cell
            n12 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            n12_per_cell_cuda[ bpg, tpg ](N_cells, n12, w1, w2,
                      npairs, shuffled_idx1, shuffled_idx2,
					  prefix_sum_pair, prefix_sum1, prefix_sum2)
            
            param_s = allocate_empty(npairs_tot, use_cuda, dtype=np.float64)
            param_logL = allocate_empty(npairs_tot, use_cuda, dtype=np.float64)
            
            # Process particle pairs in batches
            N_batch = int( npairs_tot  / self.batch_size ) + 1
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_batch, seed )
            bpg, tpg = cuda_tpb_bpg_1d( N_batch )
            perform_collisions_cuda[ bpg, tpg ](N_batch, 
                        self.batch_size, npairs_tot,
                        prefix_sum1, prefix_sum2,
                        shuffled_idx1, shuffled_idx2, cell_idx,
                        self.species1.density, self.species2.density,
                        self.species1.temperature, self.species2.temperature,
                        n12, m1, m2,
                        q1, q2, w1, w2,
                        ux1, uy1, uz1,
                        ux2, uy2, uz2,
                        dt, self.coulomb_log,
                        random_states, self.debug,
                        param_s, param_logL)

            if self.debug:
                N_cells_plasma = cell_idx[-1] - cell_idx[0]
                mean_T1 = cupy.sum( self.species1.temperature ) / N_cells_plasma
                mean_T2 = cupy.sum( self.species2.temperature ) / N_cells_plasma
                max_T1 = cupy.max( self.species1.temperature )
                max_T2 = cupy.max( self.species2.temperature )

                print("\n<T1> = ", (k / e) * mean_T1, " eV")
                print("<T2> = ", (k / e) * mean_T2, " eV")
                print("max(T1) = ", (k / e) * max_T1, " eV")
                print("max(T2) = ", (k / e) * max_T2, " eV")

                print("<s> = ", cupy.sum( param_s ) / npairs_tot)
                print("min(s) = ", cupy.min( param_s ))
                print("max(s) = ", cupy.max( param_s ))

                print("<logL> = ", cupy.sum( param_logL ) / npairs_tot)
                print("min(logL) = ", cupy.min( param_logL ))
                print("max(logL) = ", cupy.max( param_logL ))

            setattr(self.species1.ux, 'ux', ux1)
            setattr(self.species1.uy, 'uy', uy1)
            setattr(self.species1.uz, 'uz', uz1)
            setattr(self.species2.ux, 'ux', ux2)
            setattr(self.species2.uy, 'uy', uy2)
            setattr(self.species2.uz, 'uz', uz2)
