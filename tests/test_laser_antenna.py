# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This test file is part of FB-PIC (Fourier-Bessel Particle-In-Cell).

It tests the injection of a laser by a laser antenna

The laser is emitted from an antenna, and then its 2D profile is
compared with theory. There is typically a strong influence of the
longitudinal resolution on the amplitude of the emitted laser:
below ~30 points per laser wavelength, the emitted a0 can be ~10%
smaller than the desired value.

Usage :
-------
In order to show the images of the laser, and manually check the
agreement between the simulation and the theory:
$ python tests/test_laser_antenna.py
(except when setting show to False in the parameters below)

In order to let Python check the agreement between the curve without
having to look at the plots
$ py.test -q tests/test_laser_antenna.py
or
$ python setup.py test
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import c, m_e, e
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.openpmd_diag import FieldDiagnostic
from fbpic.lpa_utils.boosted_frame import BoostConverter

# Parameters
# ----------
show = True # Whether to show the plots, and check them manually
write_files = True
use_cuda = True

# Simulation box
Nz = 800
zmin = -10.e-6
zmax = 10.e-6
Nr = 25
rmax = 400.e-6
Nm = 2
dt = (zmax-zmin)/Nz/c
# Laser pulse
w0 = 128.e-6
ctau = 5.e-6
a0 = 1.
zf = 0.e-6
z0_antenna = 0.e-6
# Propagation
Lprop = 10.5e-6
Ntot_step = int(Lprop/(c*dt))
N_show = 5 # Number of instants in which to show the plots (during propagation)

# The boost in the case of the boosted frame run
gamma_boost = 10.

def test_antenna_labframe(show=False, write_files=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the emission of a laser by an antenna, in the lab frame
    """
    run_and_check_laser_antenna(None, show, write_files, z0=z0_antenna-ctau)

def test_antenna_labframe_moving( show=False, write_files=False ):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the emission of a laser by a moving antenna, in the lab frame
    """
    run_and_check_laser_antenna( None, show, write_files, z0=z0_antenna+ctau,
                                    v=c, forward_propagating=False )

def test_antenna_boostedframe(show=False, write_files=False):
    """
    Function that is run by py.test, when doing `python setup.py test`
    Test the emission of a laser by an antenna, in the boosted frame
    """
    run_and_check_laser_antenna(gamma_boost, show, write_files,
                                z0=z0_antenna-ctau)

def run_and_check_laser_antenna(gamma_b, show, write_files,
                            z0, v=0, forward_propagating=True ):
    """
    Generic function, which runs and check the laser antenna for
    both boosted frame and lab frame

    Parameters
    ----------
    gamma_b: float or None
        The Lorentz factor of the boosted frame

    show: bool
        Whether to show the images of the laser as pop-up windows

    write_files: bool
        Whether to output openPMD data of the laser

    v: float (m/s)
        Speed of the laser antenna
    """
    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, p_zmin=0, p_zmax=0,
                    p_rmin=0, p_rmax=0, p_nz=2, p_nr=2, p_nt=2, n_e=0.,
                    zmin=zmin, use_cuda=use_cuda, boundaries='open',
                    gamma_boost=gamma_b)

    # Remove the particles
    sim.ptcl = []

    # Add the laser
    add_laser( sim, a0, w0, ctau, z0, zf=zf, method='antenna',
        z0_antenna=z0_antenna, v_antenna=v, gamma_boost=gamma_b,
        fw_propagating=forward_propagating )

    # Calculate the number of steps between each output
    N_step = int( round( Ntot_step/N_show ) )

    # Add diagnostic
    if write_files:
        sim.diags = [
            FieldDiagnostic( N_step, sim.fld, comm=None,
                             fieldtypes=["rho", "E", "B", "J"] )
            ]

    # Loop over the iterations
    print('Running the simulation...')
    for it in range(N_show) :
        print( 'Diagnostic point %d/%d' %(it, N_show) )
        # Advance the Maxwell equations
        sim.step( N_step, show_progress=False )
        # Plot the fields during the simulation
        if show==True:
            show_fields( sim.fld.interp[1], 'Er' )
    # Finish the remaining iterations
    sim.step( Ntot_step - N_show*N_step, show_progress=False )

    # Check the transverse E and B field
    Nz_half = int(sim.fld.interp[1].Nz/2) + 2
    z = sim.fld.interp[1].z[Nz_half:-(sim.comm.n_guard+sim.comm.nz_damp+\
                            sim.comm.n_inject)]
    r = sim.fld.interp[1].r
    # Loop through the different fields
    for fieldtype, info_in_real_part, factor in [ ('Er', True, 2.), \
                ('Et', False, 2.), ('Br', False, 2.*c), ('Bt', True, 2.*c) ]:
        # factor correspond to the factor that has to be applied
        # in order to get a value which is comparable to an electric field
        # (Because of the definition of the interpolation grid, the )
        field = getattr(sim.fld.interp[1], fieldtype)\
                            [Nz_half:-(sim.comm.n_guard+sim.comm.nz_damp+\
                             sim.comm.n_inject)]
        print( 'Checking %s' %fieldtype )
        check_fields( factor*field, z, r, info_in_real_part,
                        z0, gamma_b, forward_propagating )
        print( 'OK' )

def check_fields( interp1_complex, z, r, info_in_real_part, z0, gamma_b,
                    forward_propagating, show_difference=False ):
    """
    Check the real and imaginary part of the interpolation grid agree
    with the theory by:
    - Checking that the part (real or imaginary) that does not
        carry information is zero
    - Extracting the a0 from the other part and comparing it
        to the predicted value
    - Using the extracted value of a0 to compare the simulated
      profile with a gaussian profile
    """
    # Extract the part that has information
    if info_in_real_part:
        interp1 = interp1_complex.real
        zero_part = interp1_complex.imag
    else:
        interp1 = interp1_complex.imag
        zero_part = interp1_complex.real

    # Control that the part that has no information is 0
    assert np.allclose( 0., zero_part, atol=1.e-6*interp1.max() )

    # Get the predicted properties of the laser in the boosted frame
    if gamma_b is None:
        boost = BoostConverter(1.)
    else:
        boost = BoostConverter(gamma_b)
    ctau_b, lambda0_b, Lprop_b, z0_b = \
        boost.copropag_length([ctau, 0.8e-6, Lprop, z0])
    # Take into account whether the pulse is propagating forward or backward
    if not forward_propagating:
        Lprop_b = - Lprop_b

    # Fit the on-axis profile to extract a0
    def fit_function(z, a0, z0_phase):
        return( gaussian_laser( z, r[0], a0, z0_phase,
                                z0_b+Lprop_b, ctau_b, lambda0_b ) )
    fit_result = curve_fit( fit_function, z, interp1[:,0],
                            p0=np.array([a0, z0_b+Lprop_b]) )
    a0_fit, z0_fit = fit_result[0]

    # Check that the a0 agrees within 5% of the predicted value
    assert abs( abs(a0_fit) - a0 )/a0 < 0.05

    # Calculate predicted fields
    r2d, z2d = np.meshgrid(r, z)
    # Factor 0.5 due to the definition of the interpolation grid
    interp1_predicted = gaussian_laser( z2d, r2d, a0_fit, z0_fit,
                                        z0_b+Lprop_b, ctau_b, lambda0_b )
    # Plot the difference
    if show_difference:
        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.imshow( interp1.T )
        plt.colorbar()
        plt.subplot(312)
        plt.imshow( interp1_predicted.T )
        plt.colorbar()
        plt.subplot(313)
        plt.imshow( (interp1_predicted - interp1).T )
        plt.colorbar()
        plt.show()
    # Control the values (with a precision of 3%)
    assert np.allclose( interp1_predicted, interp1, atol=3.e-2*interp1.max() )

def gaussian_laser( z, r, a0, z0_phase, z0_prop, ctau, lambda0 ):
    """
    Returns a Gaussian laser profile
    """
    k0 = 2*np.pi/lambda0
    E0 = a0*m_e*c**2*k0/e
    return( E0*np.exp( -r**2/w0**2 - (z-z0_prop)**2/ctau**2 ) \
                *np.cos( k0*(z-z0_phase) ) )


def show_fields( grid, fieldtype ):
    """
    Show the field `fieldtype` on the interpolation grid

    Parameters
    ----------
    grid: an instance of FieldInterpolationGrid
        Contains the field on the interpolation grid for
        on particular azimuthal mode

    fieldtype : string
        Name of the field to be plotted.
        (either 'Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz',
        'Jr', 'Jt', 'Jz', 'rho')
    """
    # matplotlib only needs to be imported if this function is called
    import matplotlib.pyplot as plt

    # Select the field to plot
    plotted_field = getattr( grid, fieldtype)
    # Show the field also below the axis for a more realistic picture
    plotted_field = np.hstack( (plotted_field[:,::-1],plotted_field) )
    extent = 1.e6*np.array([grid.zmin, grid.zmax, -grid.rmax, grid.rmax])
    plt.clf()
    plt.suptitle('%s, for mode %d' %(fieldtype, grid.m) )

    # Plot the real part
    plt.subplot(211)
    plt.imshow( plotted_field.real.T[::-1], aspect='auto',
                interpolation='nearest', extent=extent )
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Real part')

    # Plot the imaginary part
    plt.subplot(212)
    plt.imshow( plotted_field.imag.T[::-1], aspect='auto',
                interpolation='nearest', extent = extent )
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Imaginary part')

    plt.show()



if __name__ == '__main__' :

    # Run the testing functions
    test_antenna_labframe(show, write_files)
    test_antenna_labframe_moving(show, write_files)
    test_antenna_boostedframe(show, write_files)
