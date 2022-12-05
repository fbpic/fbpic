# Copyright 2017, FBPIC contributors
# Authors: Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
Thermalization test
"""

# -------
# Imports
# -------
import shutil, math
import numpy as np
from scipy.constants import c, m_e, e, k
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.openpmd_diag import ParticleDiagnostic
from fbpic.fields.smoothing import BinomialSmoother
from fbpic.particles.elementary_process.collisions import MCCollisions
# Import openPMD-viewer for checking output files
from openpmd_viewer import OpenPMDTimeSeries

# ----------
# Parameters
# ----------
use_cuda = True

def run_simulation():
    """
    Run a simulation with a uniform plasma that undergoes
    electron-ion collisions
    """
    import matplotlib.pyplot as plt
    for _ in range(3):
        # The simulation box
        Nr = 8           # Number of gridpoints along z
        Nz = 16          # Number of gridpoints along r

        zmax = 1.e-5 * Nz
        zmin = 0.e-6
        rmax = 1.e-5 * Nr
        
        Nm = 2            # Number of modes used

        # Plasma properties
        n_i = 1e28
        n_e = 1e28
        Te  = 102
        Ti  = 92

        m_i = 10*m_e

        uth_i = math.sqrt(e * Ti / (m_i * c**2))  # Normalized thermal momenta
        uth_e = math.sqrt(e * Te / (m_e * c**2))

        p_nz = 30         # Number of particles per cell along z
        p_nr = 30         # Number of particles per cell along r
        p_nt = 4          # Number of particles per cell along theta

        # Collision parameters
        coulomb_log = 5.0
        collision_period = 1
        start_collision = 1

        # Simulation time
        Time = 70.e-15
        dt = 0.7e-15
        #dt = (zmax / Nz) / c

        # Number of timesteps
        N_step = int(Time / dt)

        # The diagnostics
        diag_period = 5 # Period of the diagnostics in number of timesteps

        # Initialize the simulation object, with the neutralizing electrons
        # No particles are created because we do not pass the density
        sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
            exchange_period=100000,
            boundaries={'z':'open', 'r':'reflective'}, use_cuda=use_cuda,
            smoother=BinomialSmoother(7, False))

        # Add the charge-neutralizing electrons
        elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
                            ux_th=uth_e, uy_th=uth_e, uz_th=uth_e,
                            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                            continuous_injection=False )
        # Add the N atoms
        ions = sim.add_new_species( q=+e, m=m_i, n=n_i,
                            ux_th=uth_i, uy_th=uth_i, uz_th=uth_i,
                            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                            continuous_injection=False )

        sim.collisions = [
            MCCollisions(elec, ions, use_cuda,
                    coulomb_log = coulomb_log,
                    start = start_collision,
                    period = collision_period,
                    debug = False),
            MCCollisions(elec, elec, use_cuda,
                    coulomb_log = 1000.,
                    start = start_collision,
                    period = collision_period,
                    debug = False),
            MCCollisions(ions, ions, use_cuda,
                    coulomb_log = 1000.,
                    start = start_collision,
                    period = collision_period,
                    debug = False)
        ]

        # Add a particle diagnostic
        sim.diags = [ ParticleDiagnostic( diag_period, {"elec":elec, "ions":ions},
            write_dir='tests/diags', comm=sim.comm) ]

        # Run the simulation
        sim.step( N_step, move_positions=False )


        # Check consistency in the regular openPMD diagnostics
        ts = OpenPMDTimeSeries('./tests/diags/hdf5/')

        Tmi = np.zeros(len(ts.iterations))
        Tme = np.zeros(len(ts.iterations))
        l = 0

        for i in ts.iterations:
            ux_i = ts.get_particle(['ux'], species='ions', iteration=i)
            ux_e = ts.get_particle(['ux'], species='elec', iteration=i)

            uy_i = ts.get_particle(['uy'], species='ions', iteration=i)
            uy_e = ts.get_particle(['uy'], species='elec', iteration=i)

            uz_i = ts.get_particle(['uz'], species='ions', iteration=i)
            uz_e = ts.get_particle(['uz'], species='elec', iteration=i)


            Tmi[l] = calculate_temperature(m_i, ux_i[0], uy_i[0], uz_i[0])
            Tme[l] = calculate_temperature(m_e, ux_e[0], uy_e[0], uz_e[0])
            l+=1
        
        plt.plot(ts.iterations * dt * 1e15, Tmi * (k/e), '-C0o')
        plt.plot(ts.iterations * dt * 1e15, Tme * (k/e), '-C1o')

        # Remove openPMD files
        shutil.rmtree('./tests/diags/')

    # NRL relaxation	
    T_e = Te
    T_i = Ti
    t_points = 3000
    Te_theory = np.zeros(t_points)
    Ti_theory = np.zeros(t_points)
    Time_plot = ts.iterations[-1] * dt
    dtp = Time_plot / t_points
    t_array = np.linspace(0, Time_plot, t_points, endpoint=True)
    coeff = 1.8e-19 
    Zi = 1
    Ze = -1
    n_i_cgs = n_i / 1000
    m_e_cgs = 9.1094e-28
    m_i_cgs = 10*m_e_cgs
    
    for i in range(t_points):
        Te_theory[i] = T_e
        Ti_theory[i] = T_i
        nu0 = coeff * Zi**2 * Ze**2 * n_i_cgs * coulomb_log * \
            np.sqrt(m_e_cgs * m_i_cgs) / ((m_e_cgs*T_i + m_i_cgs*T_e)**(3/2)) / 1000
        T_e += nu0 * (T_i - T_e) * dtp
        T_i += nu0 * (T_e - T_i) * dtp

    plt.plot(t_array * 1e15, Ti_theory, 'k')
    plt.plot(t_array * 1e15, Te_theory, 'k')
    plt.legend(["i+", "e-"])
    plt.xlabel( 't [fs]' )
    plt.ylabel( 'T [eV]' )
    plt.grid()
    plt.title( 'Temperature of i+, e-, and NRL formula (black lines)' )
    plt.show()

def calculate_temperature( mass, ux, uy, uz ):
    Np = int(len(ux))

    vx_m = np.sum(ux)*c
    vy_m = np.sum(uy)*c
    vz_m = np.sum(uz)*c

    v2 = np.sum(ux**2 + uy**2 + uz**2)*c**2

    invNp = (1. / Np)
    v2 *= invNp
    vx_m *= invNp
    vy_m *= invNp
    vz_m *= invNp
    v2_m = vx_m**2 + vy_m**2 + vz_m**2
    vdiff = (v2 - v2_m)
    if vdiff < 0.:
        return 0.
    else:
        return (mass / (3. * k)) * vdiff

def test_collision_thermalization():
    run_simulation()

# Run the tests
if __name__ == '__main__':
    test_collision_thermalization()
