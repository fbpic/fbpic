# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the PsatdCoeffs class.
"""
import numpy as np
from scipy.constants import c, mu_0, epsilon_0
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import cupy


class PsatdCoeffs(object) :
    """
    Contains the coefficients of the PSATD scheme for a given mode.
    """

    def __init__( self, kz, kr, m, dt, Nz, Nr, V=None,
                  use_galilean=False, use_cuda=False ) :
        """
        Allocates the coefficients matrices for the psatd scheme.

        Parameters
        ----------
        kz : 2darray of float
            The positions of the longitudinal, spectral grid

        kr : 2darray of float
            The positions of the radial, spectral grid

        m : int
            Index of the mode

        dt : float
            The timestep of the simulation

        V: float or None, optional
            If this variable is None, the standard PSATD is used (default).
            Otherwise, the current is assumed to be "comoving",
            i.e. constant with respect to (z - v_comoving * t).
            This can be done in two ways: either by
            - Using a PSATD scheme that takes this hypothesis into account
            - Solving the PSATD scheme in a Galilean frame

        use_galilean: bool, optional
            Determines which one of the two above schemes is used
            When use_galilean is true, the whole grid moves
            with a speed v_comoving

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Shortcuts
        i = 1.j

        # Register m and dt
        self.m = m
        self.dt = dt
        inv_dt = 1./dt
        # Register velocity of galilean/comoving frame
        self.V = V

        # Construct the omega and inverse omega array
        w = c*np.sqrt( kz**2 + kr**2 )
        inv_w = 1./np.where( w == 0, 1., w ) # Avoid division by 0

        # Construct the C coefficient arrays
        self.C = np.cos( w*dt )
        # Construct the S/w coefficient arrays
        self.S_w = np.sin( w*dt )*inv_w
        # Enforce the right value for w==0
        self.S_w[ w==0 ] = dt

        # Calculate coefficients that are specific to galilean/comoving scheme
        if self.V is not None:

            # Theta coefficients due to galilean/comoving scheme
            T2 = np.exp(i*kz*V*dt)
            if use_galilean is False:
                T = np.exp(i*0.5*kz*V*dt)
            # The coefficients T_cc and T_eb abstract the modification
            # of the comoving current or galilean frame, so that the Maxwell
            # equations can be written in the same form
            if use_galilean:
                self.T_eb = T2
                self.T_cc = np.ones_like(T2)
            else:
                self.T_cc = T
                self.T_eb = np.ones_like(T2)

            # Theta-like coefficient for calculation of rho_diff
            if V != 0.:
                i_kz_V = i*kz*self.V
                i_kz_V[ kz==0 ] = 1.
                self.T_rho = np.where(
                    kz == 0., -self.dt, (1.-T2)/(self.T_cc*i_kz_V) )
            else:
                self.T_rho = -self.dt*np.ones_like(kz)

            # Precalculate some coefficients
            if V != 0.:
                # Calculate pre-factor
                inv_w_kzV = 1./np.where(
                                (w**2 - kz**2 * V**2)==0,
                                1.,
                                (w**2 - kz**2 * V**2) )
                # Calculate factor involving 1/T2
                inv_1_T2 = 1./np.where(T2 == 1, 1., 1-T2)
                # Calculate Xi 1 coefficient
                xi_1 = 1./self.T_cc * inv_w_kzV \
                       * (1. - T2*self.C + i*kz*V*T2*self.S_w)
                # Calculate Xi 2 coefficient
                xi_2 = np.where(
                        kz!=0,
                        inv_w_kzV * ( 1. \
                            + i*kz*V * T2 * self.S_w * inv_1_T2 \
                            + kz**2*V**2 * inv_w**2 * T2 * \
                            inv_1_T2*(1-self.C) ),
                        1.*inv_w**2 * (1.-self.S_w*inv_dt) )
                # Calculate Xi 3 coefficient
                xi_3 = np.where(
                        kz!=0,
                        self.T_eb * inv_w_kzV * ( self.C \
                            + i*kz*V * T2 *self.S_w * inv_1_T2 \
                            + kz**2*V**2 * inv_w**2 * \
                            inv_1_T2 * (1-self.C) ),
                        1.*inv_w**2 * (self.C-self.S_w*inv_dt) )

            # Calculate correction coefficient for j
            if V !=0:
                self.j_corr_coef = np.where( kz != 0,
                            (-i*kz*V)*inv_1_T2,
                            inv_dt )
            else:
                self.j_corr_coef = inv_dt*np.ones_like(kz)

        # Construct j_coef array (for use in the Maxwell equations)
        if V is None or V == 0:
            self.j_coef = mu_0*c**2*(1.-self.C)*inv_w**2
        else:
            self.j_coef = mu_0*c**2*(xi_1)
        # Enforce the right value for w==0
        self.j_coef[ w==0 ] = mu_0*c**2*(0.5*dt**2)

        # Calculate rho_prev coefficient array
        if V is None or V == 0:
            self.rho_prev_coef = c**2/epsilon_0*(
                self.C - inv_dt*self.S_w )*inv_w**2
        else:
            self.rho_prev_coef = c**2/epsilon_0*(xi_3)
        # Enforce the right value for w==0
        self.rho_prev_coef[ w==0 ] = c**2/epsilon_0*(-1./3*dt**2)

        # Calculate rho_next coefficient array
        if V is None or V == 0:
            self.rho_next_coef = c**2/epsilon_0*(
                1 - inv_dt*self.S_w )*inv_w**2
        else:
            self.rho_next_coef = c**2/epsilon_0*(xi_2)
        # Enforce the right value for w==0
        self.rho_next_coef[ w==0 ] = c**2/epsilon_0*(1./6*dt**2)

        # Replace these array by arrays on the GPU, when using cuda
        if use_cuda:
            self.d_C = cupy.asarray(self.C)
            self.d_S_w = cupy.asarray(self.S_w)
            self.d_j_coef = cupy.asarray(self.j_coef)
            self.d_rho_prev_coef = cupy.asarray(self.rho_prev_coef)
            self.d_rho_next_coef = cupy.asarray(self.rho_next_coef)
            if self.V is not None:
                # Variables which are specific to the Galilean/comoving scheme
                self.d_T_eb = cupy.asarray(self.T_eb)
                self.d_T_cc = cupy.asarray(self.T_cc)
                self.d_T_rho = cupy.asarray(self.T_rho)
                self.d_j_corr_coef = cupy.asarray(self.j_corr_coef)
