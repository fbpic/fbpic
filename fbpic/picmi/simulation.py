# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines the picmi Simulation interface
"""
from scipy.constants import c

# Import relevant fbpic object
from fbpic.main import Simulation as FBPICSimulation
from fbpic.fields.smoothing import BinomialSmoother
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser

# Import picmi base class
from picmistandard import PICMI_Simulation
from picmistandard import PICMI_CylindricalGrid
from picmistandard import PICMI_LaserAntenna, PICMI_GaussianLaser

# Define a new simulation object for picmi, that derives from PICMI_Simulation
class Simulation( PICMI_Simulation ):

    # Redefine the `init` method, as required by the picmi `_ClassWithInit`
    def init(self, kw):

        # Get the grid
        grid = self.solver.grid
        if not type(grid) == PICMI_CylindricalGrid:
            raise ValueError('When using fbpic with PICMI, '
                'the grid needs to be a CylindricalGrid object.')
        # Check rmin and boundary conditions
        assert grid.rmin == 0.
        assert grid.bc_zmin == grid.bc_zmax
        assert grid.bc_zmax in ['periodic', 'open']
        assert grid.bc_rmax == 'reflective'

        # TODO: Check that the solver is EM / PSATD

        # Determine timestep
        if self.solver.cfl is not None:
            dz = (grid.zmax-grid.zmin)/grid.nz
            dt = self.solver.cfl * dz / c
        elif self.time_step_size is not None:
            dt = self.time_step_size
        else:
            raise ValueError(
                'You need to either set the `cfl` of the solver\n'
                'or the `timestep_size` of the `Simulation`.')

        # Convert API for the smoother
        if self.solver.source_smoother is None:
            smoother = BinomialSmoother()
        else:
            smoother = BinomialSmoother(
                n_passes=self.solver.source_smoother.n_pass,
                compensator=self.solver.source_smoother.compensation )

        # Initialize and store the FBPIC simulation object
        self.fbpic_sim = FBPICSimulation(
            Nz=int(grid.nz), zmin=grid.zmin, zmax=grid.zmax,
            Nr=int(grid.nr), rmax=grid.rmax, Nm=grid.n_azimuthal_modes,
            dt=dt, use_cuda=True, boundaries=grid.bc_zmax,
            smoother=smoother )
        # Remove default electron species
        self.fbpic_sim.ptcl = []
        # Set the moving window
        if grid.moving_window_velocity is not None:
            self.fbpic_sim.set_moving_window(grid.moving_window_velocity[-1])


    # Redefine the method `add_laser` from the PICMI Simulation class
    def add_laser( self, laser, injection_method ):
        # Call method of parent class
        PICMI_Simulation.add_laser( self, laser, injection_method )

        # Handle injection method
        assert type(injection_method) == PICMI_LaserAntenna
        # Handle laser profile method
        if type(laser) == PICMI_GaussianLaser:
            assert laser.propagation_direction[0] == 0.
            assert laser.propagation_direction[1] == 0.
            assert (laser.zeta is None) or (laser.zeta == 0)
            assert (laser.beta is None) or (laser.beta == 0)
            laser_profile = GaussianLaser( a0=laser.a0, waist=laser.waist,
                z0=laser.centroid_position[-1], zf=laser.focal_position[-1],
                tau=laser.duration, theta_pol=laser.polarization_angle,
                phi2_chirp=laser.phi2 )
        else:
            raise ValueError('Unknown laser profile: %s' %type(injection_method))

        # Inject the laser
        add_laser_pulse( self.fbpic_sim, laser_profile, method='antenna',
            z0_antenna=injection_method.position[-1] )
