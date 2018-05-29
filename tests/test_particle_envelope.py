import numpy as np
from scipy.constants import c, e, m_e, pi
from scipy.optimize import curve_fit
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
w0 = 4.e-6
ctau = 5.e-6
k0 = 2*np.pi/0.8e-6
a0 = 0.001
# Propagation
L_prop = 30.e-6
zf = 25.e-6
N_diag = 200 # Number of diagnostic points along the propagation
# Checking the results
N_show = 2
rtol = 1.e-4

p_zmin = 15.e-6
dz = (zmax - zmin)/Nz
p_zmax = p_zmin + dz
p_rmin = 0
p_rmax = 40.e-6
n_e = 1.
dens_func = None

boundaries = 'open'
v_comoving = 0
use_galilean = False
v_window = c
dt = (zmax-zmin)*1./c/Nz*1
m = 0
Nm = abs(m)+1

def init_fields( sim, w, ctau, k0, z0, zf, a0, m=1 ) :
    """
    Imprints the appropriate profile on the fields of the simulation.

    Parameters
    ----------
    sim: Simulation object from fbpic

    w : float
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
        profile = GaussianLaser( a0=a0, waist=w, tau=tau,
                    lambda0=lambda0, z0=z0, zf=zf )
    elif m == 1 or m == -1:
        profile = LaguerreGaussLaser( 0, 1, a0=a0, waist=w, tau=tau,
                    lambda0=lambda0, z0=z0, zf=zf )

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


sim = Simulation( Nz, zmax, Nr, Lr, Nm, dt, p_zmin=p_zmin, p_zmax=p_zmax,
                p_rmin=p_rmin, p_rmax=p_rmax, p_nz=1, p_nr=1, p_nt=1, n_e=n_e,
                n_order=n_order, zmin=zmin, use_cuda=use_cuda,
                boundaries=boundaries, v_comoving=v_comoving,
                exchange_period = 1, use_galilean=use_galilean,
                use_envelope = True )

# Remove the particles
sim.ptcl = []

sim.add_new_species( q=-e, m=m_e, n=n_e, dens_func=dens_func,
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
N_step = int( round( Ntot_step/N_diag ) )
for it in range(N_diag) :
    sim.step( N_step, show_progress= False )


show_fields(sim.fld.envelope_interp[m], 'a')
show_fields(sim.fld.envelope_interp[m], 'a_old')
import matplotlib.pyplot as plt



radial_distance = np.sqrt(sim.ptcl[0].x**2 + sim.ptcl[0].y**2)
radial_momentum = np.sqrt(sim.ptcl[0].ux**2 + sim.ptcl[0].uy**2)

plt.plot(radial_distance, radial_momentum, label='Simulated')

def radial_momentum_profile(r, A, sigma):
    return A * r * np.exp(-r**2/(2*sigma**2))

# For mode 0:
z_particles = 0.5 * (p_zmax + p_zmin)
ZR = 0.5*k0*w0**2
waist = w0 * np.sqrt(1 + ((z_particles - zf)/ZR)**2 )
print((1 + ((z_particles - zf)/ZR)**2 ))
Amplitude = 0.5 * np.sqrt(2*pi) * ctau * a0**2 / waist**2 / (1 + ((z_particles - zf)/ZR)**2 )
Sigma = waist * 0.5
r = radial_distance[0:10]
u = radial_momentum[0:10]
fit_result = curve_fit(radial_momentum_profile, r,
                    u, p0=np.array([Amplitude, Sigma]) )
A, S = fit_result[0]
print('Amplitude expected and calculated', Amplitude, A)
print('Sigma expected and calculated', Sigma, S)
plt.plot(radial_distance, radial_momentum_profile(radial_distance, Amplitude, Sigma), label = 'Theoretical')
print('r', radial_distance)
print('theoretical momentum', radial_momentum_profile(radial_distance, Amplitude, Sigma))
print('momentum', radial_momentum)
print('ratio',1/radial_momentum_profile(radial_distance, Amplitude, Sigma) * radial_momentum )
plt.show()
