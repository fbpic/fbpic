"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the injection of a laser by a laser antenna

Usage :
-------
In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
$ python tests/test_fields.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_laser_antenna.py
or
$ python setup.py test
"""
import matplotlib.pyplot as plt
from scipy.constants import c
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser

# Parameters
# ----------
show = True  # Whether to show the plots, and check them manually

use_cuda = True

# Simulation box
Nz = 200
zmin = -10.e-6
zmax = 10.e-6
Nr = 25
rmax = 20.e-6
Nm = 2
dt = (zmax-zmin)/Nz/c
# Laser pulse
w0 = 4.e-6
ctau = 5.e-6
a0 = 1.
z0_antenna = 0.e-6
zf = 0.e-6
z0 = - 10.e-6
# Propagation
Lprop = 20.e-6
Ntot_step = int(Lprop/(c*dt))
N_show = 3 # Number of instants in which to show the plots (during propagation)

def test_laser_antenna(show=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the emission of a laser by an antenna
    """
    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, p_zmin=0, p_zmax=0,
                    p_rmin=0, p_rmax=0, p_nz=2, p_nr=2, p_nt=2, n_e=0.,
                    zmin=zmin, use_cuda=use_cuda, boundaries='open' )

    # Remove the particles
    sim.ptcl = []

    # Add the laser
    add_laser( sim, a0, w0, ctau, z0, zf=zf,
        method='antenna', z0_antenna=z0_antenna )

    # Calculate the number of steps between each output
    N_step = int( round( Ntot_step/N_show ) )

    # Loop over the iterations
    print('Running the simulation...')
    for it in range(N_show) :
        print 'Diagnostic point %d/%d' %(it, N_show)
        # Plot the fields during the simulation
        if show==True :
            plt.clf()
            sim.fld.interp[1].show('Et')
            plt.show()
        # Advance the Maxwell equations
        sim.step( N_step, show_progress=False )

if __name__ == '__main__' :

    # Run the testing function
    test_laser_antenna(show=show)
