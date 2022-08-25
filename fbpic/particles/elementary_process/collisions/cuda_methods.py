# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in Compton scattering (on GPU).
"""
import math as m
from numba import cuda
from scipy.constants import c, k, h, epsilon_0, m_e
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import compile_cupy
from numba.cuda.random import xoroshiro128p_uniform_float64


@compile_cupy
def pairs_per_cell_cuda(N_batch, array_in1, array_in2, array_out):
    """
    Calculate the number of pairs per cell
    """
    i = cuda.grid(1)
    if i < N_batch:
        if array_in1[i] == 0 or array_in2[i] == 0:
            array_out[i] = 0
        else:
            if array_in1[i] > array_in2[i]:
                array_out[i] = array_in1[i]
            else:
                array_out[i] = array_in2[i]


@compile_cupy
def density_per_cell_cuda(N_batch, density, weights, npart,
                          prefix_sum, d_invvol, Nz):
    """
    Calculate density of species per cell
    """
    i = cuda.grid(1)
    if i < N_batch:
        sum = 0.
        if i > 0:
            i_min = int(prefix_sum[i-1])
        else:
            i_min = 0
        for j in range(int(npart[i])):
            sum += weights[i_min + j]
        invvol = d_invvol[int(i / Nz)]
        density[i] = sum * invvol


@compile_cupy
def n12_per_cell_cuda(N_batch, n12, w1, w2,
                      npart1, prefix_sum1, prefix_sum2):
    """
    Calculate n12 of species per cell
	n12 is the sum of minimum species weights
    """
    i = cuda.grid(1)
    if i < N_batch:
        if npart1[i] == 0:
            n12[i] = 0.
        else:
            if i > 0:
                i_min1 = int(prefix_sum1[i-1])
                i_min2 = int(prefix_sum2[i-1])
            else:
                i_min1 = 0
                i_min2 = 0
            sum = 0.
            for j in range(int(npart1[i])):
                if w1[i_min1+j] < w2[i_min2+j]:
                    sum += w1[i_min1+j]
                else:
                    sum += w2[i_min2+j]
                n12[i] = sum


@compile_cupy
def temperature_per_cell_cuda(N_batch, T, npart,
                              prefix_sum, mass,
                              ux, uy, uz):
	"""
	Calculate temperature of species in cell
	"""
	i = cuda.grid(1)
	if i < N_batch:
		if npart[i] == 0:
			T[i] = 0.
		else:
			if i > 0:
				i_min = int(prefix_sum[i-1])
			else:
				i_min = 0

			vx_mean = 0.
			vy_mean = 0.
			vz_mean = 0.
			v2 = 0.
			for j in range(int(npart[i])):
				vx_mean += ux[i_min + j] * c
				vy_mean += uy[i_min + j] * c
				vz_mean += uz[i_min + j] * c

				v2 += (ux[i_min + j]**2 + uy[i_min + j]**2 + uz[i_min + j]**2) * c**2

			invNp = (1. / npart[i])
			v2 *= invNp
			vx_mean *= invNp
			vy_mean *= invNp
			vz_mean *= invNp
			u_mean2 = vx_mean**2 + vy_mean**2 + vz_mean**2
			T[i] = ( mass / (3. * k ) ) * (v2 - u_mean2)


@compile_cupy
def get_cell_idx_per_pair(N_batch, cell_idx, npair, prefix_sum_pair):
    """
    Get the cell index of each pair.
    The cell index is 1d and calculated by:
    prefix sum of pairs

    Parameters
    ----------
    cell_idx : 1darray of integers
            	The cell index of the pair

    npair : 1darray of integers
            	The particle pair array

	prefix_sum_pair : 1darray of integers
				Cumulative prefix sum of
				the particle pair array		
    """
    i = cuda.grid(1)
    if i < N_batch:
        if i > 0:
            s = prefix_sum_pair[i-1]
        else:
            s = 0
        p = npair[i]
        k = 0
        while k < p:
            cell_idx[s+k] = i
            k += 1


@cuda.jit
def perform_collisions_cuda(N_batch, batch_size, npairs_tot,
                        	prefix_sum1, prefix_sum2,
                            cell_idx, npart1, npart2,
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
	# Loop over cells
	i = cuda.grid(1)
	# Loop over batch of particle pairs and perform collision
	if i < N_batch:
		# Loop through the batch
		N_max = min( (i+1)*batch_size, npairs_tot )
		for ip in range( i*batch_size, N_max ):
			cell = cell_idx[ip]

			# Note: currently this code does not perform a non-repetitive
			# shuffle of the particle pairs. This means that certain particles
			# will be randomly chosen multiple times for collisions. This can be
			# improved by performing the shuffling separately (prior) to ensure
			# that no particles are repeated. 
			if cell > 0:
				si1 = int(round(xoroshiro128p_uniform_float64(random_states, i)
								* (npart1[cell] - 1)) + prefix_sum1[cell-1])
				si2 = int(round(xoroshiro128p_uniform_float64(random_states, i)
								* (npart2[cell] - 1)) + prefix_sum2[cell-1])
			else:
				si1 = int(round(xoroshiro128p_uniform_float64(
					random_states, i) * (npart1[cell] - 1)))
				si2 = int(round(xoroshiro128p_uniform_float64(
					random_states, i) * (npart2[cell] - 1)))

			r1 = m.sqrt( x1[si1]**2 + y1[si1]**2 )

			r2 = m.sqrt( x2[si2]**2 + y2[si2]**2 )

			distance = m.sqrt( (r1 - r2)**2 + (z1[si1] - z2[si2])**2 )

			# Choose to collide pairs that are close (half of the grid size)
			if distance < 0.001:
				# Effective time interval for the collisions
				corr_dt = dt * n12[cell] / n1[cell]

				px1 = ux1[si1] * (m1 * c)
				py1 = uy1[si1] * (m1 * c) 
				pz1 = uz1[si1] * (m1 * c) 

				px2 = ux2[si2] * (m2 * c)
				py2 = uy2[si2] * (m2 * c) 
				pz2 = uz2[si2] * (m2 * c)

				inv_m1 = 1. / m1
				vx1 = px1 * inv_m1
				vy1 = py1 * inv_m1
				vz1 = pz1 * inv_m1

				inv_m2 = 1. / m2
				vx2 = px2 * inv_m2
				vy2 = py2 * inv_m2
				vz2 = pz2 * inv_m2

				# Calculate Lorentz factor
				gamma1 = 1. / m.sqrt(1. - (vx1**2 + vy1**2 + vz1**2) / c**2)
				gamma2 = 1. / m.sqrt(1. - (vx2**2 + vy2**2 + vz2**2) / c**2)

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
					Debye2 = k * T1[cell] / (4. * m.pi * n1[cell] * q1 * q1) + \
						k * T2[cell] / (4. * m.pi * n2[cell] * q2 * q2)
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
				phi = xoroshiro128p_uniform_float64(random_states, i) * 2.0 * m.pi

				# Calculate the deflection angle
				U = xoroshiro128p_uniform_float64(
					random_states, i)    # random float [0,1]
				if s12 < 4:
					a = 0.37 * s - 0.005 * s**2 - 0.0064 * s**3
					sin2X2 = a * U / m.sqrt(1 - U + a**2 * U)
					cosX = 1. - 2. * sin2X2
					sinX = 2. * m.sqrt(sin2X2 * (1.-sin2X2))
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

				U1 = xoroshiro128p_uniform_float64(
					random_states, i)    # random float [0,1]
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
