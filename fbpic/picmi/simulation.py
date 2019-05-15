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

# Import picmi base class
from picmistandard import PICMI_Simulation
from picmistandard import PICMI_CylindricalGrid

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

        # Initialize and store the FBPIC simulation object
        self.fbpic_sim = FBPICSimulation(
            Nz=int(grid.nz), zmin=grid.zmin, zmax=grid.zmax,
            Nr=int(grid.nr), rmax=grid.rmax, Nm=grid.n_azimuthal_modes,
            dt=dt, use_cuda=True, boundaries=grid.bc_zmax )
        # Remove default electron species
        self.fbpic_sim.ptcl = []
        # Set the moving window
        if grid.moving_window_velocity is not None:
            self.fbpic_sim.set_moving_window(grid.moving_window_velocity[-1])
