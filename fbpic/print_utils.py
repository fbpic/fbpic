# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file
It defines a set of generic functions for printing simulation information.
"""
import sys, time
import numba
from fbpic import __version__

def print_simulation_setup( sim, level=1 ):
    """
    Print information about the simulation.
    - Version of FBPIC
    - CPU or GPU computation
    - Number of parallel MPI domains
    - Number of threads in case of CPU multi-threading
    - (Additional detailed information)

    Parameters
    ----------
    sim: an fbpic Simulation object
        Contains all the information of the simulation setup

    level: int, optional
        Level of detail of the simulation information
        0 - Print no information
        1 (Default) - Print basic information
        2 - Print detailed information
    """
    if sim.comm.rank == 0 and level > 0:
        # Print version of FBPIC
        message = '\n' + u'\u2630' + u'\u2630' + u'\u2630'
        message += '  FBPIC (fbpic-%s)  '%__version__
        message += u'\u2630' + u'\u2630' + u'\u2630' + '\n'
        # Basic information
        if level == 1:
            # Print information about computational setup
            if sim.use_cuda:
                message += "\nRunning on GPU "
            else:
                message += "\nRunning on CPU "
            if sim.comm.size > 1:
                message += "with %d MPI processes " %sim.comm.size
            if sim.use_threading and not sim.use_cuda:
                message += "(%d threads per process) " \
                    %numba.config.NUMBA_NUM_THREADS
        # Detailed information
        if level == 2:
            print('TBD...')

        print( message )

def progression_bar( i, Ntot, avg_time_per_step, prev_time,
                     n_avg=20, Nbars=35, char=u'\u2588'):
    """
    Shows a progression bar with Nbars and the estimated
    remaining simulation time.
    """
    # Estimate average time per step
    curr_time = time.time()
    time_per_step = curr_time - prev_time
    avg_time_per_step += (time_per_step - avg_time_per_step)/n_avg
    if i <= 2:
        # Ignores first step in time estimation (compilation time)
        avg_time_per_step = time_per_step
    # Print progress bar
    if i == 0:
        # Let the user know that the first step is much longer
        sys.stdout.write('\r' + '1st iteration & ' + \
            'Just-In-Time compilation (up to one minute) ...')
        sys.stdout.flush()
    else:
        # Print the progression bar
        nbars = int( (i+1)*1./Ntot*Nbars )
        sys.stdout.write('\r' + nbars*char )
        sys.stdout.write((Nbars-nbars)*' ')
        sys.stdout.write(' %d/%d' %(i+1, Ntot))
        sys.stdout.write(', %d ms/step' %(time_per_step*1.e3))
        if i < n_avg:
            # Time estimation is only printed after n_avg timesteps
            sys.stdout.write(', calculating ETA...')
            sys.stdout.flush()
        else:
            # Estimated time in seconds until it will finish
            eta = avg_time_per_step*(Ntot-i)
            # Conversion to H:M:S
            m, s = divmod(eta, 60)
            h, m = divmod(m, 60)
            sys.stdout.write(', %d:%02d:%02d left' % (h, m, s))
            sys.stdout.flush()
    # Clear line
    sys.stdout.write('\033[K')

    return avg_time_per_step, curr_time

def print_runtime_summary( N, duration ):
    """
    Print a summary about the total runtime of the simulation.

    Parameters
    ----------
    N: int
        The total number of iterations performed by the step loop

    duration: float, seconds
        The total time taken by the step loop
    """
    avg_tps = (duration / N)*1.e3
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    print('\nTime taken (with compilation): %d:%02d:%02d' %(h, m, s))
    print('Average time per iteration ' \
          '(with compilation): %d ms\n' %(avg_tps))
