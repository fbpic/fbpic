# Copyright 2017, FBPIC contributors
# Authors: Kristoffer Lindvall
# License: 3-Clause-BSD-LBNL
"""
Isotropization test
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
        n_e = 1e28
        Te_par   = 17.
        Te_perp  = 15.2
        
        uth_e_par = math.sqrt(e * Te_par / (m_e * c**2))
        uth_e_perp = math.sqrt(e * Te_perp / (m_e * c**2))

        p_nz = 20         # Number of particles per cell along z
        p_nr = 20         # Number of particles per cell along r
        p_nt = 4          # Number of particles per cell along theta

        # Collision parameters
        coulomb_log = 2.0
        collision_period = 1
        start_collision = 1

        # Simulation time
        Time = 1.4e-15
        dt = 0.1e-16
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
                            ux_th=uth_e_perp, uy_th=uth_e_perp, uz_th=uth_e_par,
                            p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                            continuous_injection=False )

        sim.collisions = [
            MCCollisions(elec, elec, use_cuda,
                    coulomb_log = coulomb_log,
                    start = start_collision,
                    period = collision_period,
                    debug = False),
        ]

        # Add a particle diagnostic
        sim.diags = [ ParticleDiagnostic( diag_period, {"elec":elec},
            write_dir='tests/diags', comm=sim.comm) ]

        # Run the simulation
        sim.step( N_step, move_positions=False )


        # Check consistency in the regular openPMD diagnostics
        ts = OpenPMDTimeSeries('./tests/diags/hdf5/')

        Tme_par = np.zeros(len(ts.iterations))
        Tme_perp = np.zeros(len(ts.iterations))
        l = 0

        for i in ts.iterations:
            ux_e = ts.get_particle(['ux'], species='elec', iteration=i)
            uy_e = ts.get_particle(['uy'], species='elec', iteration=i)
            uz_e = ts.get_particle(['uz'], species='elec', iteration=i)

            Tme_par[l] = calculate_par_temperature(m_e, uz_e[0])
            Tme_perp[l] = calculate_perp_temperature(m_e, ux_e[0], uy_e[0])
            l+=1

        plt.plot(ts.iterations * dt * 1e15, Tme_par * (k/e), '-C0o')
        plt.plot(ts.iterations * dt * 1e15, Tme_perp * (k/e), '-C1o')

        # Remove openPMD files
        shutil.rmtree('./tests/diags/')
    
    # NRL relaxation	
    T_e_par  = Tme_par[0] * (k/e)
    T_e_perp = Tme_perp[0] * (k/e)
    t_points = 3000
    Te_par_theory = np.zeros(t_points)
    Te_perp_theory = np.zeros(t_points)
    Time_plot = ts.iterations[-1] * dt
    dtp = Time_plot / t_points
    t_array = np.linspace(0, Time_plot, t_points, endpoint=True)

    n_e_cgs = n_e / 1000
    m_e_cgs = 9.1094e-28
    k_cgs = 1.6e-12
    coeff = 2. * math.sqrt(math.pi) * e**2 * e**2 / math.sqrt(m_e_cgs)
    
    for i in range(t_points):
        Te_par_theory[i] = T_e_par
        Te_perp_theory[i] = T_e_perp
        A = T_e_perp/T_e_par - 1.
        if A > 0:
            term = np.arctanh(A**0.5)/A**0.5
        else:
            term = np.arctanh((-A)**0.5)/(-A)**0.5
        nu0 = coeff * n_e_cgs * coulomb_log / (k_cgs * T_e_par)**1.5 * A**-2 *(
            -3. + (A+3.) * term ) * 1e20

        print(nu0)

        T_e_par  -= 2. * nu0 * (T_e_par - T_e_perp) * dtp * 1e15
        T_e_perp +=      nu0 * (T_e_par - T_e_perp) * dtp * 1e15
    
    plt.plot(t_array * 1e15, Te_par_theory, 'k')
    plt.plot(t_array * 1e15, Te_perp_theory, 'k')
    plt.legend(["Te_trans", "Te_perp"])
    plt.xlabel( 't [fs]' )
    plt.ylabel( 'T [eV]' )
    plt.grid()
    plt.title( 'Temperature of e-, and NRL formula (black lines)' )
    plt.show()

def calculate_perp_temperature( mass, ux, uy ):
    Np = int(len(ux))
    inv_gamma = np.array(Np)

    inv_gamma = 1. / np.sqrt(1. + ux**2 + uy**2)

    vx_m = np.sum(ux * inv_gamma) * c 
    vy_m = np.sum(uy * inv_gamma) * c

    v2 = np.sum((ux**2 + uy**2) * inv_gamma**2)*c**2 

    invNp = (1. / Np)
    v2 *= invNp
    vx_m *= invNp
    vy_m *= invNp
    v2_m = vx_m**2 + vy_m**2 
    vdiff = (v2 - v2_m)
    if vdiff < 0.:
        return 0.
    else:
        return (mass / (3. * k)) * vdiff / 2.0

def calculate_par_temperature( mass, uz ):
    Np = int(len(uz))
    gamma = np.array(Np)

    gamma = np.sqrt(1. + uz**2)

    vz_m = np.sum(uz/gamma)*c

    v2 = np.sum(uz**2/gamma**2)*c**2

    invNp = (1. / Np)
    v2 *= invNp
    vz_m *= invNp
    v2_m = vz_m**2
    vdiff = (v2 - v2_m)
    if vdiff < 0.:
        return 0.
    else:
        return (mass / (3. * k)) * vdiff

def test_collision_isotropization():
    run_simulation()

# Run the tests
if __name__ == '__main__':
    test_collision_isotropization()
