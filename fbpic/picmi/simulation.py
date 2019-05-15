# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines the picmi Simulation interface
"""
from scipy.constants import c, e
from .particle_charge_and_mass import particle_charge, particle_mass

# Import relevant fbpic object
from fbpic.main import Simulation as FBPICSimulation
from fbpic.fields.smoothing import BinomialSmoother
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser

# Import picmi base class
from picmistandard import PICMI_Simulation, PICMI_CylindricalGrid
from picmistandard import PICMI_AnalyticDistribution, PICMI_GriddedLayout
from picmistandard import PICMI_LaserAntenna, PICMI_GaussianLaser
from picmistandard import PICMI_Species, PICMI_MultiSpecies

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


    # Redefine the method `add_species` from the PICMI Simulation class
    def add_species( self, species, layout, initialize_self_field=False ):
        # Call method of parent class
        PICMI_Simulation.add_species( self, species, layout,
                                      initialize_self_field )

        # Extract list of species
        if type(species) == PICMI_Species:
            species_instances_list = [species]
        elif type(species) == PICMI_MultiSpecies:
            species_instances_list = species.species_instances_list
        else:
            raise ValueError('Unknown type: %s' %type(species))
        # Loop over species
        for s in species_instances_list:

            # Get their charge and mass
            if s.particle_type is not None:
                q = particle_charge[s.particle_type]
                m = particle_mass[s.particle_type]
            else:
                q = s.charge
                m = s.mass
            # If `charge_state` is set, redefine the charge
            if s.charge_state is not None:
                q = s.charge_state*e

            # Add the species to the simulation

            # - For the case of a plasma defined in a gridded layout
            if (type(s.initial_distribution)==PICMI_AnalyticDistribution) and \
                (type(layout) == PICMI_GriddedLayout):
                import numexpr
                def dens_func(z, r):
                    n = numexpr.evaluate(s.initial_distribution.density_expression)
                    return n
                p_nz = layout.n_macroparticle_per_cell['z']
                p_nr = layout.n_macroparticle_per_cell['r']
                p_nt = layout.n_macroparticle_per_cell['theta']
                self.fbpic_sim.add_new_species( q=q, m=m, n=1.,
                    dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                    continuous_injection=s.initial_distribution.fill_in )

            # Register in dictionary (useful later for diagnostics)
