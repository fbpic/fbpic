"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the implementation of the ponderomotive force (for the envelope)
by driving a very weak laser pulse through a "sheet" of particles.
- The mode 0 is tested by using a linearly polarized gaussian beam,
   which propagates to the right.
- The mode 1 is tested by using a linearly-polarized Laguerre-Gauss beam

In both cases we compare the final values of the radial momentum of the
particles after the laser completely passed them with the theoretical values.

These tests are performed in 2 cases:
- Periodic box
- Moving window

Usage :
--------
In order to show the obtained radial momentum compared to the theory:
$ python tests/test_particle_envelope.py
(Set the 'show' parameter to True for the graphs to show)
"""
import numpy as np
from scipy.constants import c, e, m_e, pi
from scipy.optimize import curve_fit
from scipy.special import genlaguerre
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, \
    GaussianLaser, LaguerreGaussLaser

# Parameters
# ----------
# (See the documentation of the function propagate_pulse
# below for their definition)
show = False # Whether to show the plots, and check them manually
use_cuda = True

# Simulation box
Nz = 300
zmin = -30.e-6
zmax = 30.e-6
Nr = 50
Lr = 40.e-6
n_order = -1
# Laser pulse
w0 = 8.e-6
ctau = 5.e-6
k0 = 2*np.pi/0.8e-6
a0 = 0.001
# Propagation
L_prop = 30.e-6
zf = 25.e-6
# Checking the results
N_show = 2
rtol = 1.e-2

p_zmin = 15.e-6
dz = (zmax - zmin)/Nz
p_zmax = p_zmin + dz
p_rmin = 0
p_rmax = 40.e-6
n_e = 1.


def test_particles_periodic(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the basic movements of particles in a periodic box.
    """
    # Choose the regular timestep, not an arbitrarily long timestep
    # (because the envelope uses a_old, which might not fit inside the window)
    dt = (zmax-zmin)*1./c/Nz
    # Test modes up to m=1
    for m in [0, 1]:

        print('')
        print('Testing mode m=%d' %m)
        propagate_pulse( Nz, Nr, abs(m)+1, zmin, zmax, Lr, L_prop, zf, dt,
                        p_zmin, p_zmax, p_rmin, p_rmax, n_e,
                        w0, ctau, k0, a0, m, N_show, n_order,
                        rtol, boundaries='periodic', v_window=0, show=show )

    print('')

def test_particles_moving_window(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the basic movements of particles in a moving window
    """
    # Choose the regular timestep (required by moving window)
    dt = (zmax-zmin)*1./c/Nz
    # Test modes up to m=1
    for m in [0, 1]:

        print('')
        print('Testing mode m=%d' %m)
        propagate_pulse( Nz, Nr, abs(m)+1, zmin, zmax, Lr, L_prop, zf, dt,
                        p_zmin, p_zmax, p_rmin, p_rmax, n_e,
                        w0, ctau, k0, a0, m, N_show, n_order,
                        rtol, boundaries='open', v_window=c, show=show )

    print('')

def propagate_pulse( Nz, Nr, Nm, zmin, zmax, Lr, L_prop, zf, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, n_e,
        w0, ctau, k0, a0, m, N_show, n_order, rtol,
        boundaries, v_window=0, use_galilean=False, v_comoving=0, show=False ):
        """
        Propagate the beam over a distance L_prop in Nt steps,
        and extracts the radial momentum of each particle at the end.

        Parameters
        ----------
        show : bool
           Whether to show the momentum, so that the user can manually check
           the agreement with the theory.

        Nz, Nr : int
           The number of points on the grid in z and r respectively

        Nm : int
            The number of modes in the azimuthal direction

        zmin, zmax : float
            The limits of the box in z

        Lr : float
           The size of the box in the r direction
           (In the case of Lr, this is the distance from the *axis*
           to the outer boundary)

        L_prop : float
           The total propagation distance (in meters)

        zf : float
           The position of the focal plane of the laser (only works for m=1)

        dt : float
           The timestep of the simulation

        p_zmin, p_zmax, p_rmin, p_rmax: float (in meters)
            The positions delimiting the particles original positions

        n_e : float (in m^-3)
            The density of electrons in the plasma, set very low so that the
            field created by the movement of electrons can be safely neglected
            in the theory

        w0 : float
           The initial waist of the laser (in meters)

        ctau : float
           The initial temporal waist of the laser (in meters)

        k0 : float
           The central wavevector of the laser (in meters^-1)

        a0 : float
           The initial a0 of the pulse

        m : int
           Index of the mode to be tested
           For m = 1 : test with a gaussian, linearly polarized beam
           For m = 0 : test with an annular beam, polarized in E_theta

        n_order : int
           Order of the stencil

        rtol : float
           Relative precision with which the results are tested

        boundaries : string
            Type of boundary condition
            Either 'open' or 'periodic'

        v_window : float
            Speed of the moving window

        v_comoving : float
            Velocity at which the currents are assumed to move

        use_galilean: bool
            Whether to use a galilean frame that moves at the speed v_comoving

        Returns
        -------
        A dictionary containing :
        - 'a' : 1d array containing the values of the amplitude
        - 'w' : 1d array containing the values of waist
        - 'fld' : the Fields object at the end of the simulation.
        """
        # Initialize the simulation object
        sim = Simulation( Nz, zmax, Nr, Lr, Nm, dt, n_order=n_order, zmin=zmin,
                        use_cuda=use_cuda, boundaries=boundaries,
                        v_comoving=v_comoving, exchange_period=1,
                        use_galilean=use_galilean, use_envelope=True )
        sim.ptcl = []  # Remove the empty electron species

        # Add our thin "sheet" of electrons
        sim.ptcl = []
        sim.add_new_species( q=-e, m=m_e, n=n_e,
                              p_nz=1, p_nr=1, p_nt=1,
                              p_zmin=p_zmin, p_zmax=p_zmax,
                              p_rmin=p_rmin, p_rmax=p_rmax,
                              continuous_injection = False)
        if v_window !=0:
            sim.set_moving_window( v=v_window )

        # Initialize the laser fields
        z0 = (zmax+zmin)/2
        init_fields( sim, w0, ctau, k0, z0, zf, a0, m )

        # Calculate the number of steps to run between each diagnostic
        Ntot_step = int( round( L_prop/(c*dt) ) )
        sim.step( Ntot_step, show_progress= False )

        # Physical quantities
        z_particles = 0.5 * (p_zmax + p_zmin)
        ZR = 0.5*k0*w0**2
        # The waist is calculated at the position of the particles
        w = w0 * np.sqrt(1 + ((z_particles - zf)/ZR)**2 )
        A = 0.5 * np.sqrt(2*pi) * ctau * a0**2 \
                        / (1 + ((z_particles - zf)/ZR)**2 )
        if m == 1:
            # Taking into account the scale factor from LaguerreGaussLaser when
            # m is not 0
            A *= 2

        # Calculate the radial momentum of particles
        radial_distance = np.sqrt(sim.ptcl[0].x**2 + sim.ptcl[0].y**2)
        radial_momentum = (sim.ptcl[0].ux*sim.ptcl[0].x + \
                        sim.ptcl[0].uy*sim.ptcl[0].y) / radial_distance
        if m == 1:
            # The pulse is at pi/4
            cos_theta = (sim.ptcl[0].x + sim.ptcl[0].y)/2**.5 /radial_distance

        if show:
            import matplotlib.pyplot as plt
            if m == 0:
                plt.plot(1e6*radial_distance, radial_momentum,
                    'o', label='Simulated')
                plt.plot(1e6*radial_distance,
                    radial_momentum_profile_gaussian(radial_distance, A, w),
                    '--', label = 'Theoretical')
            else:
                plt.plot(1e6*radial_distance, radial_momentum/cos_theta**2,
                    'o', label='Simulated')
                plt.plot(1e6*radial_distance,
                    radial_momentum_profile_laguerre(radial_distance, A, w),
                    '--', label = 'Theoretical')
            plt.xlabel('r (microns)')
            plt.ylabel('ur(kg.m/s)')
            plt.title('Radial momentum')
            plt.legend(loc=0)
            plt.show()
        # Check the accuracy of the results
        if m == 0:
            assert np.allclose( radial_momentum,
                radial_momentum_profile_gaussian(radial_distance, A, w),
                atol=rtol*radial_momentum.max() )
        elif m == 1:
            assert np.allclose( radial_momentum/cos_theta**2,
                radial_momentum_profile_laguerre(radial_distance, A, w),
                atol=rtol*radial_momentum.max() )
        print('The simulation results agree with the theory to %e.' %rtol)

def init_fields( sim, w0, ctau, k0, z0, zf, a0, m=1 ) :
    """
    Imprints the appropriate profile on the fields of the simulation.

    Parameters
    ----------
    sim: Simulation object from fbpic

    w0 : float
       The initial waist of the laser (in microns)

    ctau : float
       The initial temporal waist of the laser (in microns)

    k0 : flat
       The central wavevector of the laser (in microns^-1)

    z0 : float
       The position of the centroid on the z axis

    zf : float
       The position of the focal plane

    a0 : float
       The initial a0 of the pulse

    m: int, optional
        The mode on which to imprint the profile
    """
    # Initialize the fields
    tau = ctau/c
    lambda0 = 2*np.pi/k0
    # Create the relevant laser profile
    if m == 0:
        profile = GaussianLaser( a0=a0, waist=w0, tau=tau,
                    lambda0=lambda0, z0=z0, zf=zf )
    elif m == 1:
        # Put the peak of the Laguerre-Gauss at pi/4 to check that the
        # angular dependency is correctly captured
        profile = LaguerreGaussLaser( 0, 1, a0, w0, tau,
                    z0, lambda0=lambda0, zf=zf, theta0=np.pi/4 )

    # Add the profiles to the simulation
    add_laser_pulse( sim, profile, method = 'direct_envelope' )


def radial_momentum_profile_gaussian(r, A, w):
    """
    Calculte the transverse profile for mode 0

    This is used for the fit of the momentum

    Parameters
    ----------
    r: 1darray
       Represents the positions of the grid in r

    w : float
       The waist of the laser at the position of the particles

    A : float
       The amplitude of the momentum profile
    """
    return A * r/w**2 * np.exp(-2*r**2/(w**2))


def radial_momentum_profile_laguerre(r, A, w):
    """
    Calculte the transverse profile for mode 1

    This is used for the fit of the momentum

    Parameters
    ----------
    r: 1darray
       Represents the positions of the grid in r

    w : float
       The waist of the laser at the position of the particles

    A : float
       The amplitude of the momentum profile
    """
    # We create the relevant Laguerre polynomial, which is actually
    # Constant equal to one, but is written down
    # for the sake of clarity and consistency
    laguerre_pol = genlaguerre(0,1)
    return A*r**2/w**2*(2*r/w**2 - 1/r) * laguerre_pol(2*r**2/w**2)**2 \
                * np.exp(-2*r**2/(w**2))


if __name__ == '__main__' :

    # Run the testing function
    test_particles_periodic(show=show)

    test_particles_moving_window(show=show)
