# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FBPIC)
It defines the class that performs calculation of Monte-Carlo collisions.
"""
import numpy as np
import random
import math as m
import cupy
from scipy.constants import c, k, epsilon_0, e

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
        get_cell_idx_per_pair_cuda, get_shuffled_idx_per_particle_cuda, \
        alt_n12_per_cell_cuda

class MCCollisions(object):
    """
    Simulate Monte-Carlo (MC) collisions. The calculations randomly
    pair all particles in a cell and perform collisions
    """
    def __init__( self, species1, species2, use_cuda=False,
                coulomb_log = 0.,
                start = 0,
                collision_period = 0 ):
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
        self.collision_period = collision_period

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

    def test_perform_collisions_cuda(self, N_batch, batch_size, npairs_tot,
                        	prefix_sum1, prefix_sum2,
                            shuffled_idx1, shuffled_idx2, cell_idx,
                            n1, n2, n12, m1, m2,
                            q1, q2, w1, w2,
                            ux1, uy1, uz1,
                            ux2, uy2, uz2,
							x1, y1, z1,
                        	x2, y2, z2,
                            dt, coulomb_log,
                            T1, T2,
                            random_states):
        """
        Perform collisions between all pairs in each cell
        """
        # Loop over batch of particle pairs and perform collision
        for i in range (N_batch):
            # Loop through the batch
            N_max = min( (i+1)*batch_size, npairs_tot )
            for ip in range( i*batch_size, N_max ):
                cell = cell_idx[ip]

                if cell > 0:
                    si1 = int( shuffled_idx1[ip] + prefix_sum1[cell-1] )
                    si2 = int( shuffled_idx2[ip] + prefix_sum2[cell-1] )
                else:
                    si1 = int( shuffled_idx1[ip] )
                    si2 = int( shuffled_idx2[ip] )

                r1 = m.sqrt( x1[si1]**2 + y1[si1]**2 )

                r2 = m.sqrt( x2[si2]**2 + y2[si2]**2 )

                distance = m.sqrt( (r1 - r2)**2 + (z1[si1] - z2[si2])**2 )
                
                Debye2 = (epsilon_0 * k / e**2) / (n1[cell] / T1[cell] + n1[cell] / T2[cell])

                # Choose to collide pairs that are within a Debye sphere
                #if distance < m.sqrt(Debye2):
                if distance < 0.1:
                    # Effective time interval for the collisions
                    corr_dt = dt * n12[cell] / n1[cell]

                    px1 = ux1[si1] * (m1 * c)
                    py1 = uy1[si1] * (m1 * c) 
                    pz1 = uz1[si1] * (m1 * c)

                    print("px1", px1)
                    print("py1", py1)
                    print("pz1", pz1)


                    px2 = ux2[si2] * (m2 * c)
                    py2 = uy2[si2] * (m2 * c) 
                    pz2 = uz2[si2] * (m2 * c)

                    print("px2", px2)
                    print("py2", py2)
                    print("pz2", pz2)

                    inv_m1 = 1. / m1
                    vx1 = px1 * inv_m1
                    vy1 = py1 * inv_m1
                    vz1 = pz1 * inv_m1

                    inv_m2 = 1. / m2
                    vx2 = px2 * inv_m2
                    vy2 = py2 * inv_m2
                    vz2 = pz2 * inv_m2

                    print("vx1", vx1)
                    print("vy1", vy1)
                    print("vz1", vz1)

                    print("vx2", vx2)
                    print("vy2", vy2)
                    print("vz2", vz2)

                    print((vx1**2 + vy1**2 + vz1**2) / c**2)
                    print((vx2**2 + vy2**2 + vz2**2) / c**2)

                    # Calculate Lorentz factor
                    gamma1 = m.sqrt(1. + (vx1**2 + vy1**2 + vz1**2) / c**2)
                    gamma2 = m.sqrt(1. + (vx2**2 + vy2**2 + vz2**2) / c**2)


                    g12 = m1 * gamma1 + m2 * gamma2
                    inv_g12 = 1. / g12

                    gamma12 = m1 * gamma1 * m2 * gamma2
                    invgamma12 = 1. / gamma12

                    # Center of mass (COM) velocity
                    COM_vx = (px1 + px2) * inv_g12
                    COM_vy = (py1 + py2) * inv_g12
                    COM_vz = (pz1 + pz2) * inv_g12

                    COM_v_v1 = (COM_vx * vx1 + COM_vy * vy1 + COM_vz * vz1)
                    COM_v_v2 = (COM_vx * vx2 + COM_vy * vy2 + COM_vz * vz2)

                    COM_v2 = COM_vx**2 + COM_vy**2 + COM_vz**2
                    COM_gamma = 1. / m.sqrt(1. - COM_v2 / c**2)

                    # momenta in COM
                    px_COM = px1 + ((COM_gamma - 1.) * COM_v_v1 / COM_v2
                                            - COM_gamma) * m1 * gamma1 * COM_vx
                    py_COM = py1 + ((COM_gamma - 1.) * COM_v_v1 / COM_v2
                                            - COM_gamma) * m1 * gamma1 * COM_vy
                    pz_COM = pz1 + ((COM_gamma - 1.) * COM_v_v1 / COM_v2
                                            - COM_gamma) * m1 * gamma1 * COM_vz

                    p_COM = m.sqrt(px_COM**2 + py_COM**2 + pz_COM**2)
                    invp_COM = 1. / p_COM
                    invp_COM2 = 1. / p_COM**2

                    # Lorentz transforms
                    gamma1_COM = (1 - COM_v_v1 / c**2) * COM_gamma * gamma1
                    gamma2_COM = (1 - COM_v_v2 / c**2) * COM_gamma * gamma2

                    # Calculate coulomb log if not user-defined
                    qq = q1 * q2
                    qq2 = qq**2
                    if coulomb_log <= 0.:
                        coeff = 1. / (4 * m.pi * epsilon_0 * c**2)
                        b0 = coeff * qq * COM_gamma * inv_g12 * \
                            (m1 * gamma1_COM * m2 * gamma2_COM * invp_COM2 * c**2 + 1.)**2
                        bmin = max(0.5 * h * invp_COM, b0)
                        coulomb_log = 0.5 * m.log(1. + Debye2 / bmin**2)
                        if coulomb_log < 2.:
                            coulomb_log = 2.

                    term1 = n1[cell] * n2[cell] / n12[cell]
                    term2 = corr_dt * coulomb_log * qq2 * \
                        invgamma12 / (4. * m.pi * epsilon_0**2 * c**4)
                    term3 = gamma2_COM * p_COM * inv_g12 * \
                        (m1 * gamma1_COM * m2 * gamma2_COM * invp_COM2 * c**2 + 1.)**2

                    # Calculate the collision parameter s12
                    s12 = term1 * term2 * term3

                    # Low temperature correction
                    v_rel = m.sqrt((vx1 - vx2)**2
                                + (vy1 - vy2)**2
                                + (vz1 - vz2)**2)
                    s_prime = (4.*m.pi/3)**(1/3) * term1 * corr_dt * \
                        ((m1 + m2) / max(m1 * n1[cell]**(2/3), m2 * n2[cell]**(2/3))) * \
                        v_rel

                    s = min(s12, s_prime)

                    # Random azimuthal angle
                    phi = random.random()*2.0 * m.pi
                    #phi = xoroshiro128p_uniform_float64(random_states, i) * 2.0 * m.pi

                    # Calculate the deflection angle
                    U = random.random()
                    #U = xoroshiro128p_uniform_float64(
                    #    random_states, i)    # random float [0,1]
                    if s12 < 4:
                        a = 0.37 * s - 0.005 * s**2 - 0.0064 * s**3
                        sin2X2 = a * U / m.sqrt(1 - U + a**2 * U)
                        cosX = 1. - 2. * sin2X2
                        sinX = 2. * m.sqrt(sin2X2 * (1. - sin2X2))
                    else:
                        cosX = 2. * U - 1.
                        sinX = m.sqrt(1. - cosX * cosX)
                    sinXcosPhi = sinX * m.cos(phi)
                    sinXsinPhi = sinX * m.sin(phi)

                    p_perp_COM = m.sqrt(px_COM**2 + py_COM**2)
                    invp = 1. / p_perp_COM

                    # Resulting momentum in COM frame
                    if (p_perp_COM > 1.e-10 * p_COM):
                        pxf1_COM = (px_COM * pz_COM * invp * sinXcosPhi
                                    - py_COM * p_COM * invp * sinXsinPhi
                                    + px_COM * cosX)
                        pyf1_COM = (py_COM * pz_COM * invp * sinXcosPhi
                                    + px_COM * p_COM * invp * sinXsinPhi
                                    + py_COM * cosX)
                        pzf1_COM = (-p_perp_COM * sinXcosPhi
                                    + pz_COM * cosX)
                    else:
                        pxf1_COM = p_COM * sinXcosPhi
                        pyf1_COM = p_COM * sinXsinPhi
                        pzf1_COM = p_COM * cosX

                    vC_pfCOM = (COM_vx * pxf1_COM
                                + COM_vy * pyf1_COM
                                + COM_vz * pzf1_COM)

                    # Resulting momentum in lab frame
                    pxf1 = pxf1_COM + COM_vx * \
                        ((COM_gamma - 1.) * vC_pfCOM / COM_v2 + m1 * gamma1_COM * COM_gamma)
                    pyf1 = pyf1_COM + COM_vy * \
                        ((COM_gamma - 1.) * vC_pfCOM / COM_v2 + m1 * gamma1_COM * COM_gamma)
                    pzf1 = pzf1_COM + COM_vz * \
                        ((COM_gamma - 1.) * vC_pfCOM / COM_v2 + m1 * gamma1_COM * COM_gamma)

                    U1 = random.random()
                    #U1 = xoroshiro128p_uniform_float64(
                    #    random_states, i)    # random float [0,1]
                    if U1 * w1[si1] < w2[si2]:
                        # Deflect particle 1
                        inv_norm1 = 1. / (m1 * c)
                        ux1[si1] = pxf1 * inv_norm1
                        uy1[si1] = pyf1 * inv_norm1
                        uz1[si1] = pzf1 * inv_norm1

                    if U1 * w2[si2] < w1[si1]:
                        # Deflect particle 2 (pf2 = -pf1)
                        inv_norm2 = 1. / (m2 * c)
                        ux2[si2] = -pxf1 * inv_norm2
                        uy2[si2] = -pyf1 * inv_norm2
                        uz2[si2] = -pzf1 * inv_norm2

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

            x1 = getattr( self.species1, 'x')
            y1 = getattr( self.species1, 'y')
            z1 = getattr( self.species1, 'z')

            x2 = getattr( self.species2, 'x')
            y2 = getattr( self.species2, 'y')
            z2 = getattr( self.species2, 'z')

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
            
            # Calculate temperature in each cell
            temperature1 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            temperature_per_cell_cuda[ bpg, tpg ]( N_cells, temperature1, npart1,
                                prefix_sum1, m1,
                                ux1, uy1, uz1 )

            temperature2 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            temperature_per_cell_cuda[ bpg, tpg ]( N_cells, temperature2, npart2,
                                prefix_sum2, m2,
                                ux2, uy2, uz2 )

            # Total number of pairs
            npairs_tot = int( cupy.sum( npairs ) )
            #print("\n Total number of pairs = ", npairs_tot)

            # Cumulative prefix sum of pairs
            prefix_sum_pair = cupy.cumsum(npairs)

            # Cell index of pair
            cell_idx = cupy.empty(npairs_tot, dtype=np.int32)
            get_cell_idx_per_pair_cuda[ bpg, tpg ](N_cells, cell_idx, npairs, prefix_sum_pair)

            #print("\nMax npairs in cell ", cupy.max(npairs))

            # Shuffled index of particle 1
            shuffled_idx1 = cupy.empty(npairs_tot, dtype=np.int32)
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_cells, seed )
            get_shuffled_idx_per_particle_cuda[ bpg, tpg ](N_cells, shuffled_idx1, npart1,
                                                            npairs, prefix_sum_pair, random_states)

            # Shuffled index of particle 2
            shuffled_idx2 = cupy.empty(npairs_tot, dtype=np.int32)
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_cells, seed )
            get_shuffled_idx_per_particle_cuda[ bpg, tpg ](N_cells, shuffled_idx2, npart2,
                                                            npairs, prefix_sum_pair, random_states)

            # Calculate sum of minimum weights in each cell
            n12 = allocate_empty(N_cells, use_cuda, dtype=np.float64)
            #n12_per_cell_cuda[ bpg, tpg ]( N_cells, n12, w1, w2, npart1,
            #                                prefix_sum1, prefix_sum2 )

            alt_n12_per_cell_cuda[ bpg, tpg ]( N_cells, n12, w1, w2,
                      npairs, shuffled_idx1, shuffled_idx2,
					  prefix_sum_pair, prefix_sum1, prefix_sum2 )

            """
            print("\nMax shuffled particle1 in cell: ", cupy.max(shuffled_idx1))
            print("\nMax shuffled particle2 in cell: ", cupy.max(shuffled_idx2))

            print("\nMax particle1 in cell: ", cupy.max(npart1))
            print("\nMax particle2 in cell: ", cupy.max(npart2))
            """
            
            # Diagnostics
            N_cells_plasma = cell_idx[-1] - cell_idx[0]
            mean_T1 = cupy.sum( temperature1 ) / N_cells_plasma
            mean_T2 = cupy.sum( temperature2 ) / N_cells_plasma
            mean_n1 = cupy.sum( n1 ) / N_cells_plasma
            mean_n2 = cupy.sum( n2 ) / N_cells_plasma
            mean_n12 = cupy.sum( n12 ) / N_cells_plasma

            mean_Debye = m.sqrt( (epsilon_0 * k / e**2 ) / 
                        ( mean_n1 / mean_T1 + mean_n1 / mean_T2) ) 

            print("\n <Debye> = ", mean_Debye)
            print("<T1> = ", (k / e) * mean_T1, " eV")
            print("<T2> = ", (k / e) * mean_T2, " eV")
            print("<n1> = ", mean_n1)
            print("<n2> = ", mean_n2)
            print("<n12> = ", mean_n12)

            # The particles need to be shuffled, in each cell, in non-repeating order
            # so that particles are randomly paired once
            
            # Process particle pairs in batches
            N_batch = int( npairs_tot  / self.batch_size ) + 1
            seed = np.random.randint( 256 )
            random_states = create_xoroshiro128p_states( N_batch, seed )
            """
            print("shuffled_idx1: ", cupy.max(shuffled_idx1))
            print("shuffled_idx2: ", cupy.max(shuffled_idx2))
            print("cell_idx: ", cupy.max(cell_idx))
            print("temperature1: ", cupy.max(temperature1))
            print("temperature2: ", cupy.max(temperature2))
            print("n1: ", cupy.max(n1))
            print("n2: ", cupy.max(n2))
            print("n12: ", cupy.max(n12))
            print("ux1: ", cupy.max(ux1))
            print("uy1: ", cupy.max(uy1))
            print("uz1: ", cupy.max(uz1))
            print("ux2: ", cupy.max(ux2))
            print("uy2: ", cupy.max(uy2))
            print("uz2: ", cupy.max(uz2))
            """
            bpg, tpg = cuda_tpb_bpg_1d( N_batch )
            perform_collisions_cuda[ bpg, tpg ]( N_batch, 
                        self.batch_size, npairs_tot,
                        prefix_sum1, prefix_sum2,
                        shuffled_idx1, shuffled_idx2, cell_idx,
                        n1, n2, n12, m1, m2,
                        q1, q2, w1, w2,
                        ux1, uy1, uz1,
                        ux2, uy2, uz2,
                        x1, y1, z1,
                        x2, y2, z2,
                        dt, self.coulomb_log,
                        temperature1, temperature2,
                        random_states )
            
            """
            self.test_perform_collisions_cuda( N_batch, 
                        self.batch_size, npairs_tot,
                        prefix_sum1, prefix_sum2,
                        shuffled_idx1, shuffled_idx2, cell_idx,
                        n1, n2, n12, m1, m2,
                        q1, q2, w1, w2,
                        ux1, uy1, uz1,
                        ux2, uy2, uz2,
                        x1, y1, z1,
                        x2, y2, z2,
                        dt, self.coulomb_log,
                        temperature1, temperature2,
                        random_states )

            print("sol ux1: ", cupy.max(ux1))
            print("sol uy1: ", cupy.max(uy1))
            print("sol uz1: ", cupy.max(uz1))
            print("sol ux2: ", cupy.max(ux2))
            print("sol uy2: ", cupy.max(uy2))
            print("sol uz2: ", cupy.max(uz2))
            """
            setattr(self.species1.ux, 'ux', ux1)
            setattr(self.species1.uy, 'uy', uy1)
            setattr(self.species1.uz, 'uz', uz1)
            setattr(self.species2.ux, 'ux', ux2)
            setattr(self.species2.uy, 'uy', uy2)
            setattr(self.species2.uz, 'uz', uz2)
