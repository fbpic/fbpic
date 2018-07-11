import numpy as np
from scipy.constants import c, mu_0, m_e, e
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
import matplotlib.pyplot as plt


# Parameters
# ----------
# (See the documentation of the function propagate_pulse
# below for their definition)
show = False # Whether to show the plots, and check them manually

use_cuda = False

# Simulation box
Nz = 175
zmin = -25.e-6
zmax = 10.e-6
Nr = 60
rmax = 6
n_order = -1
# Laser pulse
w0 = 2
k0 = 2*np.pi/0.8e-6
kp = k0 / 20
ctau = np.sqrt(2)/kp
print(ctau)

a0 = 1.5
# Propagation
L_prop = 1500/kp
zf = 25.e-6
print(L_prop)

# The particles
n_critical = k0**2 * m_e / (mu_0 * e**2) # Theoretical critical density
p_zmin = -15.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 500.e-6 # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 6  # Maximal radial position of the plasma (meters)
n_e = n_critical /400 # Density (electrons.meters^-3)
print(n_e)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r

# The diagnostics and the checkpoints/restarts
diag_period = 200         # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 50   # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint


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

def show_transform( grid, fieldtype ):
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
    #plotted_field = np.hstack( (plotted_field[:,::-1],plotted_field) )
    #extent = 1.e6*np.array([grid.kz[Nz//2-1,0], grid.kz[Nz//2 +1,0], grid.kr[0,0], grid.kr[-1,-1]])
    plt.clf()
    plt.suptitle('%s, for mode %d' %(fieldtype, grid.m) )

    # Plot the real part
    plt.subplot(211)
    plt.imshow( plotted_field.real.T[::-1], aspect='auto',
                #interpolation='nearest', extent=extent )
                interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Real part')

    # Plot the imaginary part
    plt.subplot(212)
    plt.imshow( plotted_field.imag.T[::-1], aspect='auto',
                #interpolation='nearest', extent = extent )
                interpolation='nearest')
    plt.xlabel('z')
    plt.ylabel('r')
    cb = plt.colorbar()
    cb.set_label('Imaginary part')

    plt.show()


Nm = 1
dt = (zmax-zmin)*1./c/Nz
dt = 0.075e-6/c
print(c*dt)
print(L_prop)
print(L_prop / c / dt)


sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
    p_zmin=p_zmin, p_zmax=p_zmax, p_rmin=p_rmin, p_rmax=p_rmax, p_nz=p_nz,
    p_nr=p_nr, p_nt=Nm+1, n_e=n_e, n_order=n_order, zmin=zmin,
    boundaries='open', v_comoving=0.999*c, use_galilean=True, initialize_ions=True,
    use_cuda=use_cuda, use_envelope=True )
sim.set_moving_window(v=c)
tau = ctau/c
lambda0 = 2*np.pi/k0
# Create the relevant laser profile
z0 = 0
sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm, fieldtypes = ["a", "rho", "E"] ),
        ParticleDiagnostic(diag_period, {'electrons': sim.ptcl[0]}, sim.comm )]

profile = GaussianLaser( a0=a0, waist=w0, tau=tau,
            lambda0=lambda0, z0=z0, zf=zf )

add_laser_pulse( sim, profile, method = 'direct_envelope' )



Ntot_step_init = int( round( L_prop/(c*dt) ) )
k_iter = 1
kz = sim.fld.envelope_spect[0].kz
kr = sim.fld.envelope_spect[0].kr
#show_fields(sim.fld.envelope_interp[0], 'a')

#plt.plot(sim.fld.envelope_interp[0].a.real[:,0])
#plt.plot(sim.fld.envelope_interp[0].a.imag[:,0])
#plt.show()
#show_coefs2(sim.fld.envelope_spect[0], 'a', sim.fld.psatd[0])

for it in range(k_iter):
    sim.step( Ntot_step_init//k_iter, show_progress=show)
    if show:
    show_fields(sim.fld.envelope_interp[0], 'a')
    show_fields(sim.fld.interp[0], 'rho')
    show_fields(sim.fld.envelope_interp[0], 'chi')
    show_transform(sim.fld.envelope_spect[0], 'a')
    i, j = np.unravel_index(np.argmax(np.abs(sim.fld.envelope_spect[0].a)), np.abs(sim.fld.envelope_spect[0].a).shape)
    print(i,j)
    print(kz[i][j], kr[i][j])
    print(abs(sim.fld.envelope_spect[0].a[i][j]))
    show_coefs2(sim.fld.envelope_spect[0], 'a', sim.fld.psatd[0])
    plt.plot(np.abs(sim.fld.envelope_interp[0].a[:,0]))
    plt.show()
    plt.plot(np.abs(sim.fld.envelope_interp[0].a[:,0]))
    plt.plot(sim.fld.envelope_interp[0].a.real[:,0])
    plt.plot(sim.fld.envelope_interp[0].a.imag[:,0])
    plt.show()
