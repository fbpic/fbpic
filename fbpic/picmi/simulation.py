# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines the picmi Simulation interface
"""
import numpy as np
import warnings
from scipy.constants import c, e, m_e
from .particle_charge_and_mass import particle_charge, particle_mass

# Import relevant fbpic object
from fbpic.main import Simulation as FBPICSimulation
from fbpic.fields.smoothing import BinomialSmoother
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian, add_particle_bunch
from fbpic.lpa_utils.mirrors import Mirror
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
    ParticleChargeDensityDiagnostic, \
    BackTransformedFieldDiagnostic, BackTransformedParticleDiagnostic

# Import picmi base class
from picmistandard import PICMI_Simulation, PICMI_CylindricalGrid
from picmistandard import PICMI_AnalyticDistribution, PICMI_UniformDistribution, PICMI_GriddedLayout
from picmistandard import PICMI_PseudoRandomLayout, PICMI_GaussianBunchDistribution
from picmistandard import PICMI_LaserAntenna, PICMI_GaussianLaser
from picmistandard import PICMI_Species, PICMI_MultiSpecies
from picmistandard import PICMI_FieldIonization
from picmistandard import PICMI_AnalyticAppliedField, PICMI_ConstantAppliedField, PICMI_Mirror
from picmistandard import PICMI_FieldDiagnostic, PICMI_ParticleDiagnostic, \
    PICMI_LabFrameFieldDiagnostic, PICMI_LabFrameParticleDiagnostic

# Define a new simulation object for picmi, that derives from PICMI_Simulation
class Simulation( PICMI_Simulation ):

    # Redefine the `init` method, as required by the picmi `_ClassWithInit`
    def init(self, kw):

        self.sim_kw = {}
        for argname in ['use_ruyten_shapes', 'use_modified_volume']:
            if f'fbpic_{argname}' in kw:
                self.sim_kw[argname] = kw.pop(f'fbpic_{argname}')

        self.step_kw = {}
        for argname in ['correct_currents',
                        'correct_divE',
                        'use_true_rho',
                        'move_positions',
                        'move_momenta',
                        'show_progress']:
            if f'fbpic_{argname}' in kw:
                self.step_kw[argname] = kw.pop(f'fbpic_{argname}')

        # Get the grid
        grid = self.solver.grid
        if not type(grid) == PICMI_CylindricalGrid:
            raise ValueError('When using fbpic with PICMI, '
                'the grid needs to be a CylindricalGrid object.')
        # Check rmin and boundary conditions
        assert grid.lower_bound[0] == 0.
        assert grid.lower_boundary_conditions[1] == grid.upper_boundary_conditions[1]
        if grid.lower_boundary_conditions[1] == 'reflective':
            warnings.warn(
            "FBPIC does not support reflective boundary condition in z.\n"
            "The z boundary condition was automatically converted to 'open'.")
            grid.lower_boundary_conditions[1] = 'open'
            grid.upper_boundary_conditions[1] = 'open'
        assert grid.upper_boundary_conditions[1] in ['periodic', 'open']
        assert grid.upper_boundary_conditions[0] in ['reflective', 'open']

        # Determine timestep
        if self.solver.cfl is not None:
            dz = (grid.upper_bound[1]-grid.lower_bound[1])/grid.number_of_cells[1]
            dr = (grid.upper_bound[0]-grid.lower_bound[0])/grid.number_of_cells[0]
            if self.gamma_boost is not None:
                beta = np.sqrt(1. - 1./self.gamma_boost**2)
                dr = dr/((1+beta)*self.gamma_boost)
            dt = self.solver.cfl * min(dz, dr) / c
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
            if self.solver.source_smoother.n_pass is None:
                n_passes = 1
            else:
                n_passes = {'r': self.solver.source_smoother.n_pass[0],
                            'z': self.solver.source_smoother.n_pass[1]}
            if self.solver.source_smoother.compensation is None:
                compensator = False
            else:
                compensator = all(self.solver.source_smoother.compensation)
            smoother = BinomialSmoother( n_passes=n_passes,
                                         compensator=compensator )

        # Convert verbose level:
        verbose_level = self.verbose
        if verbose_level is None:
            verbose_level = 1

        # Order of the stencil for z derivatives in the Maxwell solver
        if self.solver.stencil_order is None:
            n_order = -1
        else:
            n_order = self.solver.stencil_order[-1]

        # Number of guard cells
        if grid.guard_cells is None:
            n_guard = None
        else:
            n_guard = grid.guard_cells[-1]

        if self.solver.galilean_velocity is None:
            v_comoving = None
        else:
            v_comoving = self.solver.galilean_velocity[-1]

        # Initialize and store the FBPIC simulation object
        self.fbpic_sim = FBPICSimulation(
            Nz=int(grid.number_of_cells[1]), zmin=grid.lower_bound[1], zmax=grid.upper_bound[1],
            Nr=int(grid.number_of_cells[0]), rmax=grid.upper_bound[0], Nm=grid.n_azimuthal_modes,
            dt=dt, use_cuda=True, smoother=smoother, n_order=n_order,
            boundaries={'z':grid.upper_boundary_conditions[1], 'r':grid.upper_boundary_conditions[0]},
            n_guard=n_guard, verbose_level=verbose_level,
            particle_shape=self.particle_shape,
            v_comoving=v_comoving,
            gamma_boost=self.gamma_boost,
            **self.sim_kw)

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
            phi2_chirp = laser.phi2
            if phi2_chirp is None:
                phi2_chirp = 0
            polarization_angle = np.arctan2(laser.polarization_direction[1],
                                            laser.polarization_direction[0])
            laser_profile = GaussianLaser( a0=laser.a0, waist=laser.waist,
                z0=laser.centroid_position[-1], zf=laser.focal_position[-1],
                tau=laser.duration, theta_pol=polarization_angle,
                phi2_chirp=phi2_chirp )
        else:
            raise ValueError('Unknown laser profile: %s' %type(injection_method))

        # Inject the laser
        add_laser_pulse( self.fbpic_sim, laser_profile, method='antenna',
            z0_antenna=injection_method.position[-1],
            gamma_boost=self.gamma_boost )


    # Redefine the method `add_species` from the PICMI Simulation class
    def add_species( self, species, layout, initialize_self_field=False ):
        # Call method of parent class
        PICMI_Simulation.add_species( self, species, layout,
                                      initialize_self_field )
        # Call generic method internally
        self._add_species_generic( species, layout,
            injection_plane_position=None, injection_plane_normal_vector=None,
            initialize_self_field=initialize_self_field )


    def add_species_through_plane( self, species, layout,
            injection_plane_position, injection_plane_normal_vector,
            initialize_self_field=False ):
        # Call method of parent class
        PICMI_Simulation.add_species_through_plane( self, species, layout,
            injection_plane_position, injection_plane_normal_vector,
            initialize_self_field=initialize_self_field )
        # Call generic method internally
        self._add_species_generic( species, layout,
            injection_plane_position=injection_plane_position,
            injection_plane_normal_vector=injection_plane_normal_vector,
            initialize_self_field=initialize_self_field )


    def _add_species_generic( self, species, layout, injection_plane_position,
        injection_plane_normal_vector, initialize_self_field ):

        # Extract list of species
        if type(species) == PICMI_Species:
            species_instances_list = [species]
        elif type(species) == PICMI_MultiSpecies:
            species_instances_list = species.species_instances_list
        else:
            raise ValueError('Unknown type: %s' %type(species))

        # Loop over species and create FBPIC species
        for s in species_instances_list:

            # Get their charge and mass
            if s.particle_type is not None:
                s.charge = particle_charge[s.particle_type]
                s.mass = particle_mass[s.particle_type]
            # If `charge_state` is set, redefine the charge and mass
            if s.charge_state is not None:
                s.charge = s.charge_state*e
                s.mass -= s.charge_state*m_e

            # Add the species to the FBPIC simulation
            fbpic_species = self._create_new_fbpic_species(s,
                 layout, injection_plane_position,
                 injection_plane_normal_vector, initialize_self_field)

            # Register a pointer to the FBPIC species in the PICMI species itself
            # (Useful for particle diagnostics later on)
            s.fbpic_species = fbpic_species

        # Loop over interactions
        for interaction in self.interactions:
            assert type(interaction) is PICMI_FieldIonization
            assert interaction.model == 'ADK'
            picmi_target = interaction.product_species
            picmi_source = interaction.ionized_species
            if not hasattr( picmi_target, 'fbpic_species' ):
                raise RuntimeError('For ionization with PICMI+FBPIC:\n'
                    'You need to add the target species to the simulation,'
                    ' before the other species.')
            fbpic_target = picmi_target.fbpic_species
            fbpic_source = picmi_source.fbpic_species
            fbpic_source.make_ionizable( element=picmi_source.particle_type,
                                         level_start=picmi_source.charge_state,
                                         target_species=fbpic_target )


    def _create_new_fbpic_species(self, s, layout, injection_plane_position,
        injection_plane_normal_vector, initialize_self_field):

        # - For the case of a plasma/beam defined in a gridded layout
        if type(layout) == PICMI_GriddedLayout:
            # - Uniform distribution
            if type(s.initial_distribution)==PICMI_UniformDistribution:
                n0 = s.initial_distribution.density
                dens_func = None
            # - Analytic distribution
            elif type(s.initial_distribution)==PICMI_AnalyticDistribution:
                import numexpr
                density_expression = s.initial_distribution.density_expression
                if s.density_scale is not None:
                    n0 = s.density_scale
                else:
                    n0 = 1.
                def dens_func(x, y, z):
                    d = locals()
                    d.update( s.initial_distribution.user_defined_kw )
                    n = numexpr.evaluate( density_expression, local_dict=d )
                    return n
            else:
                raise ValueError('Unknown combination of layout and distribution')
            p_nr = layout.n_macroparticle_per_cell[0]
            p_nt = layout.n_macroparticle_per_cell[1]
            p_nz = layout.n_macroparticle_per_cell[2]

            if initialize_self_field or (injection_plane_position is not None):
                assert s.initial_distribution.fill_in != True

                if injection_plane_position is None:
                    z_injection_plane = None
                else:
                    z_injection_plane = injection_plane_position[-1]
                gamma0_beta0 = s.initial_distribution.directed_velocity[-1]/c
                gamma0 = ( 1 + gamma0_beta0**2 )**.5
                dist = s.initial_distribution
                fbpic_species = add_particle_bunch( self.fbpic_sim,
                    q=s.charge, m=s.mass, gamma0=gamma0, n=n0,
                    dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                    p_zmin=dist.lower_bound[-1] if dist.lower_bound[-1] is not None else -np.inf,
                    p_zmax=dist.upper_bound[-1] if dist.upper_bound[-1] is not None else +np.inf,
                    p_rmin=0,
                    p_rmax=dist.upper_bound[0] if dist.upper_bound[0] is not None else +np.inf,
                    boost=self.fbpic_sim.boost,
                    z_injection_plane=z_injection_plane,
                    initialize_self_field=initialize_self_field,
                    boost_positions_in_dens_func=True )
            else:
                dist = s.initial_distribution
                fbpic_species = self.fbpic_sim.add_new_species(
                    q=s.charge, m=s.mass, n=n0,
                    dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                    p_zmin=dist.lower_bound[-1] if dist.lower_bound[-1] is not None else -np.inf,
                    p_zmax=dist.upper_bound[-1] if dist.upper_bound[-1] is not None else +np.inf,
                    p_rmax=dist.upper_bound[0] if dist.upper_bound[0] is not None else +np.inf,
                    continuous_injection=s.initial_distribution.fill_in,
                    boost_positions_in_dens_func=True )

        # - For the case of a Gaussian beam
        elif (type(s.initial_distribution)==PICMI_GaussianBunchDistribution) \
             and (type(layout) == PICMI_PseudoRandomLayout):
            dist = s.initial_distribution
            gamma0_beta0 = dist.centroid_velocity[-1]/c
            gamma0 = ( 1 + gamma0_beta0**2 )**.5
            sig_r = dist.rms_bunch_size[0]
            sig_z = dist.rms_bunch_size[-1]
            sig_gamma = dist.rms_velocity[-1]/c
            sig_vr = dist.rms_velocity[0] / gamma0
            if sig_vr != 0:
                tf = - sig_r**2/sig_vr**2 * dist.velocity_divergence[0]
            else:
                tf = 0.
            zf = dist.centroid_position[-1] + \
                 dist.centroid_velocity[-1]/gamma0 * tf
            # Calculate size at focus and emittance
            sig_r0 = (sig_r**2 - (sig_vr*tf)**2)**0.5
            n_emit = gamma0 * sig_r0 * sig_vr/c
            # Get the number of physical particles
            n_physical_particles = dist.n_physical_particles
            if s.density_scale is not None:
                n_physical_particles *= s.density_scale
            fbpic_species = add_particle_bunch_gaussian( self.fbpic_sim,
                                q=s.charge, m=s.mass,
                                gamma0=gamma0, sig_gamma=sig_gamma,
                                sig_r=sig_r0, sig_z=sig_z, n_emit=n_emit,
                                n_physical_particles=n_physical_particles,
                                n_macroparticles=layout.n_macroparticles,
                                zf=zf, tf=tf, boost=self.fbpic_sim.boost,
                                initialize_self_field=initialize_self_field )

        # - For the case of an empty species
        elif (s.initial_distribution is None) and (layout is None):
            fbpic_species = self.fbpic_sim.add_new_species(q=s.charge, m=s.mass)

        else:
            raise ValueError('Unknown combination of layout and distribution')

        return fbpic_species


    # Redefine the method `add_diagnostic` of the parent class
    def add_diagnostic(self, diagnostic):
        # Call method of parent class
        PICMI_Simulation.add_diagnostic( self, diagnostic )

        # Handle iteration_min/max in regular diagnostic
        if type(diagnostic) in [PICMI_FieldDiagnostic, PICMI_ParticleDiagnostic]:
            if diagnostic.step_min is None:
                iteration_min = 0
            else:
                iteration_min = diagnostic.step_min
            if diagnostic.step_max is None:
                iteration_max = np.inf
            else:
                iteration_max = diagnostic.step_max

        # Register field diagnostic
        if type(diagnostic) in [PICMI_FieldDiagnostic, PICMI_LabFrameFieldDiagnostic]:
            if diagnostic.data_list is None:
                data_list = ['rho', 'E', 'B', 'J']
            else:
                data_list = set()  # Use set to avoid redundancy
                for data in diagnostic.data_list:
                    if data in ['Ex', 'Ey', 'Ez', 'E']:
                        data_list.add('E')
                    elif data in ['Bx', 'By', 'Bz', 'B']:
                        data_list.add('B')
                    elif data in ['Jx', 'Jy', 'Jz', 'J']:
                        data_list.add('J')
                    elif data == 'rho':
                        data_list.add('rho')
                # Use sorted to make sure that each MPI rank goes through
                # fields in the same order, when dumping to disk (esp.
                # since this operation requires an MPI gather)
                data_list = sorted(list(data_list))

        if type(diagnostic) == PICMI_FieldDiagnostic:

            diag = FieldDiagnostic(
                    period=diagnostic.period,
                    fldobject=self.fbpic_sim.fld,
                    comm=self.fbpic_sim.comm,
                    fieldtypes=data_list,
                    write_dir=diagnostic.write_dir,
                    iteration_min=iteration_min,
                    iteration_max=iteration_max)

            # Register particle density diagnostic
            rho_density_list = []
            if diagnostic.data_list is not None:
                for data in diagnostic.data_list:
                    if data.startswith('rho_'):
                        # particle density diagnostics, rho_speciesname
                        rho_density_list.append(data)
            if rho_density_list:
                species_dict = {}
                for data in rho_density_list:
                    sname = data[4:]
                    for s in self.species:
                        if s.name == sname:
                            species_dict[s.name] = s.fbpic_species
                pdd_diag = ParticleChargeDensityDiagnostic(
                            period=diagnostic.period,
                            sim=self.fbpic_sim,
                            species=species_dict,
                            write_dir=diagnostic.write_dir,
                            iteration_min=iteration_min,
                            iteration_max=iteration_max)
                self.fbpic_sim.diags.append( pdd_diag )

        elif type(diagnostic) == PICMI_LabFrameFieldDiagnostic:
            diag = BackTransformedFieldDiagnostic(
                    zmin_lab=diagnostic.grid.lower_bound[1],
                    zmax_lab=diagnostic.grid.upper_bound[1],
                    v_lab=c,
                    dt_snapshots_lab=diagnostic.dt_snapshots,
                    Ntot_snapshots_lab=diagnostic.num_snapshots,
                    gamma_boost=self.gamma_boost,
                    period=100,
                    fldobject=self.fbpic_sim.fld,
                    comm=self.fbpic_sim.comm,
                    fieldtypes=diagnostic.data_list,
                    write_dir=diagnostic.write_dir)
        # Register particle diagnostic
        elif type(diagnostic) in [PICMI_ParticleDiagnostic,
                                  PICMI_LabFrameParticleDiagnostic]:
            species_dict = {}
            for s in diagnostic.species:
                if s.name is None:
                    raise ValueError('When using a species in a diagnostic, '
                                      'its name must be set.')
                species_dict[s.name] = s.fbpic_species
            if diagnostic.data_list is None:
                data_list = ['position', 'momentum', 'weighting']
            else:
                data_list = diagnostic.data_list
            if type(diagnostic) == PICMI_ParticleDiagnostic:
                diag = ParticleDiagnostic(
                    period=diagnostic.period,
                    species=species_dict,
                    comm=self.fbpic_sim.comm,
                    particle_data=data_list,
                    write_dir=diagnostic.write_dir,
                    iteration_min=iteration_min,
                    iteration_max=iteration_max)
            else:
                diag = BackTransformedParticleDiagnostic(
                    zmin_lab=diagnostic.grid.lower_bound[1],
                    zmax_lab=diagnostic.grid.upper_bound[1],
                    v_lab=c,
                    dt_snapshots_lab=diagnostic.dt_snapshots,
                    Ntot_snapshots_lab=diagnostic.num_snapshots,
                    gamma_boost=self.gamma_boost,
                    period=100,
                    fldobject=self.fbpic_sim.fld,
                    species=species_dict,
                    comm=self.fbpic_sim.comm,
                    particle_data=data_list,
                    write_dir=diagnostic.write_dir)

        # Add it to the FBPIC simulation
        self.fbpic_sim.diags.append( diag )

    # Redefine the method `add_diagnostic` of the parent class
    def add_applied_field(self, applied_field):
        # Call method of parent class
        PICMI_Simulation.add_applied_field( self, applied_field )

        if type(applied_field) == PICMI_Mirror:
            assert applied_field.z_front_location is not None
            mirror = Mirror( z_lab=applied_field.z_front_location,
                             n_cells=applied_field.number_of_cells,
                             gamma_boost=self.fbpic_sim.boost.gamma0 )
            self.fbpic_sim.mirrors.append( mirror )

        elif type(applied_field) == PICMI_ConstantAppliedField:
            # TODO: Handle bounds
            for field_name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                field_value = getattr( applied_field, field_name )
                if field_value is None:
                    continue
                def field_func( F, x, y, z, t, amplitude, length_scale ):
                    return( F + amplitude * field_value )
                # Pass it to FBPIC
                self.fbpic_sim.external_fields.append(
                    ExternalField( field_func, field_name, 1., 0.)
                )

        elif type(applied_field) == PICMI_AnalyticAppliedField:
            # TODO: Handle bounds
            for field_name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                # Extract expression and execute it inside a function definition
                expression = getattr( applied_field, field_name+'_expression' )
                if expression is None:
                    continue
                fieldfunc = None
                define_function_code = \
                """def fieldfunc( F, x, y, z, t, amplitude, length_scale ):\n    return( F + amplitude * %s )""" %expression
                # Take into account user-defined variables
                for k in applied_field.user_defined_kw:
                    define_function_code = \
                        "%s = %s\n" %(k,applied_field.user_defined_kw[k]) \
                        + define_function_code
                exec( define_function_code, globals() )
                # Pass it to FBPIC
                self.fbpic_sim.external_fields.append(
                    ExternalField( fieldfunc, field_name, 1., 0.)
                )

        else:
            raise ValueError("Unrecognized `applied_field` type.")


    # Redefine the method `step` of the parent class
    def step(self, nsteps=None):
        if nsteps is None:
            nsteps = self.max_steps
        self.fbpic_sim.step( nsteps , **self.step_kw)
