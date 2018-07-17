import numpy as np
from scipy.constants import c, mu_0, m_e, e
from scipy.optimize import curve_fit
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, \
    GaussianLaser, LaguerreGaussLaser


# Parameters
# ----------
# (See the documentation of the function propagate_pulse
# below for their definition)
show = True # Whether to show the plots, and check them manually
if show:
    import matplotlib.pyplot as plt

use_cuda = False

# Simulation box
Nz = 300
zmin = -30.e-6
zmax = 30.e-6
Nr = 90
rmax = 45.e-6
n_order = -1
dt = 0.13e-6/c
# Laser pulse
w0 = 20.e-6
ctau = 10.e-6
k0 = 2*np.pi/0.8e-6
a0 = 0.01
# Propagation
L_prop_init = 50.e-6
L_prop_in_plasma = 50.e-6
zf = 25.e-6
# Data analysis
N_diag = 10

# The particles
n_critical = k0**2 * m_e / (mu_0 * e**2) # Theoretical critical density
p_zmin = 15.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 500.e-6 # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 40.e-6  # Maximal radial position of the plasma (meters)
n_e = n_critical * 0.05  # Density (electrons.meters^-3)
#n_e = 4.e18*1.e6
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
v_in_plasma = c * np.sqrt(1- n_e/n_critical )

# The density profile
ramp_start = 20.e-6
ramp_length = 20.e-6

def dens_func( z, r ) :
    """Returns relative density at position z and r"""
    # Allocate relative density
    n = np.ones_like(z)
    # Make linear ramp
    n = np.where( z<ramp_start+ramp_length, (z-ramp_start)/ramp_length, n )
    # Supress density before the ramp
    n = np.where( z<ramp_start, 0., n )
    return(n)

def init_fields( sim, w0, ctau, k0, zf, a0, z0=0, m=0 ) :
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
        For m = 0 : gaussian profile, linearly polarized beam
        For m = 1 : annular profile
    """
    # Initialize the fields
    tau = ctau/c
    lambda0 = 2*np.pi/k0
    # Create the relevant laser profile

    if m == 0:
        profile = GaussianLaser( a0=a0, waist=w0, tau=tau,
                    lambda0=lambda0, z0=z0, zf=zf )
    elif m == 1:
        profile = LaguerreGaussLaser( 0, 1, a0, w0, tau,
                    z0, lambda0=lambda0, zf=zf )
    # Add the profiles to the simulation
    add_laser_pulse( sim, profile, method = 'direct_envelope' )

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
                #interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Real part')

    # Plot the imaginary part
    plt.subplot(212)
    plt.imshow( plotted_field.imag.T[::-1], aspect='auto',
                interpolation='nearest', extent = extent )
                #interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Imaginary part')

    plt.show()


def longitudinal_profile(z, A, z_center):
    return A * np.exp(-(z-z_center)**2/ctau**2)

def test_for_mode(m):
    Nm = m + 1
    # Initialize the simulation
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin=p_zmin, p_zmax=p_zmax, p_rmin=p_rmin, p_rmax=p_rmax, p_nz=p_nz,
        p_nr=p_nr, p_nt=Nm+1, n_e=n_e, n_order=n_order, zmin=zmin,
        dens_func=dens_func, boundaries='open',
        use_cuda=use_cuda, use_envelope=True )

    sim.set_moving_window(v=c)

    init_fields( sim, w0, ctau, k0, zf, a0, m=Nm-1)

    Ntot_step_init = int( round( L_prop_init/(c*dt) ) )
    sim.step( Ntot_step_init, show_progress=show )

    Ntot_step = int( round( L_prop_in_plasma/(c*dt) ) )
    N_step = int( round( Ntot_step/N_diag ) )
    z_list = []
    for it in range(N_diag):
        sim.step( N_step, show_progress= False )
        z = sim.fld.envelope_interp[0].z
        profile = abs(sim.fld.envelope_interp[0].a).sum(axis=1)
        a = curve_fit(longitudinal_profile, z, profile, p0=(0.35, 0.00005 + c*N_diag*dt*(it+1)))
        A, z_center = a[0]
        z_list.append(z_center)

    time = [dt*N_step*i for i in range(N_diag)]
    vg, b = np.polyfit(time, z_list, 1)
    print(vg, c, v_in_plasma)
    if show:
        plt.plot(time, z_list)
        plt.show()

    #assert np.allclose(vg, v_in_plasma, rtol = 5e-3)

if __name__ == '__main__' :

    # Run the testing function
    test_for_mode(0)

    test_for_mode(1)
