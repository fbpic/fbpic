# Copyright 2019, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)

It defines the picmi Simulation interface
"""
import math
import numpy as np
import warnings
from typing import ClassVar
from scipy.constants import c, e, m_e
from pydantic import Field, PrivateAttr
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
from picmistandard import PICMI_Simulation, PICMI_CylindricalGrid, PICMI_ElectromagneticSolver
from picmistandard import PICMI_AnalyticDistribution, PICMI_UniformDistribution, PICMI_GriddedLayout
from picmistandard import PICMI_PseudoRandomLayout, PICMI_GaussianBunchDistribution
from picmistandard import PICMI_LaserAntenna, PICMI_GaussianLaser
from picmistandard import PICMI_Species, PICMI_MultiSpecies
from picmistandard import PICMI_FieldIonization
from picmistandard import PICMI_AnalyticAppliedField, PICMI_ConstantAppliedField, PICMI_Mirror
from picmistandard import PICMI_FieldDiagnostic, PICMI_ParticleDiagnostic, \
    PICMI_LabFrameFieldDiagnostic, PICMI_LabFrameParticleDiagnostic


def species_instances( species ):
    """
    Return the list of the individual PICMI species of `species`
    (a `MultiSpecies` object contains several species)
    """
    if isinstance( species, PICMI_MultiSpecies ):
        return list( species.species_instances_list )
    elif isinstance( species, PICMI_Species ):
        return [ species ]
    else:
        raise ValueError('Unknown type: %s' %type(species))


def fbpic_particle_shape( particle_shape ):
    """
    Return the FBPIC name of the PICMI `particle_shape`, which is given by its
    name or by the equivalent integer interpolation order (FBPIC supports
    linear and cubic shapes, and uses linear shapes by default)
    """
    fbpic_shapes = { None: 'linear',
                     'linear': 'linear', 1: 'linear',
                     'cubic': 'cubic', 3: 'cubic' }
    if isinstance( particle_shape, bool ) or \
       ( particle_shape not in fbpic_shapes ):
        raise ValueError("FBPIC only supports the particle shapes 'linear' "
            "(or 1) and 'cubic' (or 3), but got %r." %(particle_shape,))
    return fbpic_shapes[ particle_shape ]


def constant_field_func( field_value ):
    """
    Return a function that adds the constant field `field_value`,
    in the form that is expected by `ExternalField`

    (The function is created in this separate scope, so that it uses
    the value of its own field component, and not the one of the
    component that was handled last.)
    """
    def field_func( F, x, y, z, t, amplitude, length_scale ):
        return( F + amplitude * field_value )
    return field_func


class FBPICObjects:
    """
    The FBPIC objects that are created from the PICMI input of a `Simulation`
    (i.e. the state of the simulation, which is not part of its PICMI input)
    """
    def __init__( self ):
        # The FBPIC `Simulation` object
        self.sim = None
        # The FBPIC species, by PICMI species
        self.species = {}
        # The PICMI interactions that were set up in FBPIC
        self.interactions = []
        # The entries of each list of the PICMI simulation (e.g. `lasers`)
        # that were passed to FBPIC, in order
        self.created = { 'species': [], 'lasers': [],
                         'applied_fields': [], 'diagnostics': [] }
        # The PICMI input that defines the FBPIC simulation object itself,
        # when it was created (see `Simulation._simulation_input`)
        self.simulation_input = None

    # A copy of a PICMI simulation describes a new simulation, which does
    # not share (or copy) the FBPIC objects of the original simulation
    def __copy__( self ):
        return FBPICObjects()

    def __deepcopy__( self, memo ):
        return FBPICObjects()


# Define a new simulation object for picmi, that derives from PICMI_Simulation
class Simulation( PICMI_Simulation ):
    """
    In addition, the arguments below are specific to FBPIC. When they are not
    given, the corresponding default of FBPIC is used.

    The FBPIC objects (e.g. the FBPIC species) are created from the PICMI input
    when the simulation is run with `step`, and (except for the diagnostics)
    when `fbpic_sim` or `get_fbpic_species` is used. The PICMI input can thus
    be given in any order, either when creating the `Simulation` or with its
    `add_*` methods. Input that is added afterwards is passed to FBPIC the next
    time that the simulation is run, while the input that was passed to FBPIC
    cannot be changed.
    """

    # --- Arguments that are passed to the FBPIC `Simulation` object
    use_ruyten_shapes: bool | None = Field(
        default=None, alias='fbpic_use_ruyten_shapes',
        description="Whether to use Ruyten shape factors for the particle "
                    "deposition, which ensure that a uniform distribution of "
                    "macroparticles leads to a uniform charge density on the "
                    "grid, even close to the axis")
    use_modified_volume: bool | None = Field(
        default=None, alias='fbpic_use_modified_volume',
        description="Whether to use a slightly-modified, effective cell "
                    "volume, which ensures that the charge deposited near the "
                    "axis is correctly taken into account by the spectral "
                    "cylindrical Maxwell solver")

    # --- Arguments that are passed to the `step` method of the FBPIC `Simulation`
    correct_currents: bool | None = Field(
        default=None, alias='fbpic_correct_currents',
        description="Whether to correct the currents in spectral space")
    correct_divE: bool | None = Field(
        default=None, alias='fbpic_correct_divE',
        description="Whether to correct the divergence of E in spectral space")
    use_true_rho: bool | None = Field(
        default=None, alias='fbpic_use_true_rho',
        description="Whether to use the true rho deposited on the grid for "
                    "the field push or not")
    move_positions: bool | None = Field(
        default=None, alias='fbpic_move_positions',
        description="Whether to move or freeze the particles' positions")
    move_momenta: bool | None = Field(
        default=None, alias='fbpic_move_momenta',
        description="Whether to move or freeze the particles' momenta")
    show_progress: bool | None = Field(
        default=None, alias='fbpic_show_progress',
        description="Whether to show a progression bar")

    # The FBPIC-specific arguments, by the object that they are passed to
    _simulation_arguments: ClassVar[tuple[str, ...]] = (
        'use_ruyten_shapes', 'use_modified_volume' )
    _step_arguments: ClassVar[tuple[str, ...]] = (
        'correct_currents', 'correct_divE', 'use_true_rho',
        'move_positions', 'move_momenta', 'show_progress' )
    # The PICMI input that defines the FBPIC simulation object itself
    _simulation_input_names: ClassVar[tuple[str, ...]] = (
        'solver', 'time_step_size', 'verbose', 'particle_shape',
        'gamma_boost' ) + _simulation_arguments

    # The FBPIC objects that correspond to the PICMI input. (This is a private
    # attribute, since it is not part of the PICMI input itself.)
    _fbpic: FBPICObjects = PrivateAttr( default_factory=FBPICObjects )

    # Check the PICMI input, once it is validated
    def model_post_init( self, context ):
        super().model_post_init( context )
        # The FBPIC objects are only created when they are needed
        # (see `_create_fbpic_objects`), but the grid is checked already here
        self._get_grid()

    def __copy__( self ):
        # The copy does not share the FBPIC objects of this simulation
        # (`__deepcopy__` does not copy them either, see `FBPICObjects`)
        copied = super().__copy__()
        copied._fbpic = FBPICObjects()
        return copied


    @property
    def fbpic_sim( self ):
        """
        The underlying FBPIC `Simulation` object

        (Accessing it creates the FBPIC objects of the PICMI input that was
        given so far, except for the diagnostics, which are created when the
        simulation is run. With MPI, access it on all ranks, since creating
        the FBPIC objects can require communication between the ranks.)
        """
        self._create_fbpic_objects( include_diagnostics=False )
        return self._fbpic.sim


    def get_fbpic_species( self, species ):
        """
        Return the FBPIC species (`Particles` object) of a PICMI species that
        was added to the simulation, e.g. to use FBPIC features that are not
        part of PICMI (such as `track`)

        (Calling this creates the FBPIC objects of the PICMI input that was
        given so far, except for the diagnostics, which are created when the
        simulation is run. With MPI, call it on all ranks, since creating
        the FBPIC objects can require communication between the ranks.)

        Parameters
        ----------
        species: PICMI `Species` or `MultiSpecies`
            For a `MultiSpecies`, the list of the FBPIC species of its
            species is returned.
        """
        self._create_fbpic_objects( include_diagnostics=False )
        if isinstance( species, PICMI_MultiSpecies ):
            return [ self._get_fbpic_species( s )
                     for s in species_instances( species ) ]
        return self._get_fbpic_species( species )


    def _create_fbpic_objects( self, include_diagnostics=True ):
        """
        Create the FBPIC objects of the PICMI input that was not passed to
        FBPIC yet (at first, the FBPIC simulation itself)

        The objects are created in an order that does not depend on the order
        in which the PICMI input was given: the species first, then their
        interactions, and the diagnostics last (e.g. since the FBPIC particle
        diagnostics depend on whether a species is ionizable or tracked).
        """
        if self._fbpic.sim is None:
            self._create_fbpic_simulation()
            self._fbpic.simulation_input = self._simulation_input()
        else:
            changed = [ name for name, value in self._simulation_input().items()
                        if value != self._fbpic.simulation_input[name] ]
            if changed:
                raise ValueError('The input %s of the Simulation cannot be '
                    'changed once the FBPIC simulation is created (when the '
                    'simulation is run, or when `fbpic_sim` or '
                    '`get_fbpic_species` is used).'
                    %', '.join([ '`%s`' %name for name in changed ]))

        # The entries with the same index in these lists belong together
        self._check_same_length( 'species', 'layouts', 'initialize_self_fields',
            'injection_plane_positions', 'injection_plane_normal_vectors' )
        self._check_same_length( 'lasers', 'laser_injection_methods' )
        self._check_unique_species()

        self._create_new_entries( 'species',
            lambda i: self._add_species_generic( self.species[i],
                self.layouts[i],
                injection_plane_position=self.injection_plane_positions[i],
                injection_plane_normal_vector=self.injection_plane_normal_vectors[i],
                initialize_self_field=self.initialize_self_fields[i] ) )
        self._setup_interactions()
        self._create_new_entries( 'lasers',
            lambda i: self._add_fbpic_laser( self.lasers[i],
                                             self.laser_injection_methods[i] ) )
        self._create_new_entries( 'applied_fields',
            lambda i: self._add_fbpic_applied_field( self.applied_fields[i] ) )
        if include_diagnostics:
            self._create_new_entries( 'diagnostics',
                lambda i: self._add_fbpic_diagnostic( self.diagnostics[i] ) )


    def _simulation_input( self ):
        """
        Return the PICMI input that defines the FBPIC simulation object itself
        """
        return self.model_dump( include=set(self._simulation_input_names) )


    def _check_unique_species( self ):
        """
        Check that each PICMI species is added only once to the simulation
        """
        added = []
        for species in self.species:
            for s in species_instances( species ):
                if any( s is other for other in added ):
                    raise ValueError('The species %s is added more than once '
                        'to the simulation.' %(s.name or s.particle_type))
                added.append( s )


    def _check_same_length( self, *list_names ):
        """
        Check that the PICMI lists `list_names`, whose entries with the same
        index belong together, have the same length (they are filled together
        by the `add_*` methods, but they can also be given directly)
        """
        lengths = [ len( getattr(self, list_name) ) for list_name in list_names ]
        if len( set(lengths) ) > 1:
            raise ValueError('The entries with the same index in the lists %s '
                'of the Simulation belong together, but these lists have '
                'different lengths: %s.' %(', '.join(list_names), lengths))


    def _create_new_entries( self, list_name, create_entry ):
        """
        Call `create_entry(i)` for each index `i` of the PICMI list `list_name`
        (e.g. `lasers`) whose entry was not passed to FBPIC yet

        The entries that were passed to FBPIC cannot be removed or replaced,
        since their FBPIC objects exist already.
        """
        created = self._fbpic.created[list_name]
        entries = getattr( self, list_name )
        if ( len(entries) < len(created) ) or \
           any( entry is not created_entry
                for entry, created_entry in zip(entries, created) ):
            raise ValueError('The entries of `%s` that were passed to FBPIC '
                '(when the simulation was run, or when `fbpic_sim` or '
                '`get_fbpic_species` was used) cannot be removed or replaced.'
                %list_name)
        for i in range( len(created), len(entries) ):
            create_entry( i )
            created.append( entries[i] )


    def _get_grid( self ):
        """
        Return the grid of the simulation, which needs to be a CylindricalGrid
        (of an ElectromagneticSolver)
        """
        if not isinstance(self.solver, PICMI_ElectromagneticSolver):
            raise ValueError('When using fbpic with PICMI, the solver needs '
                'to be an ElectromagneticSolver object, but it is: %s'
                %type(self.solver))
        grid = self.solver.grid
        if not isinstance(grid, PICMI_CylindricalGrid):
            raise ValueError('When using fbpic with PICMI, '
                'the grid needs to be a CylindricalGrid object.')
        return grid


    def _create_fbpic_simulation( self ):
        """
        Create the FBPIC simulation object, from the PICMI simulation itself
        """
        grid = self._get_grid()
        # Check rmin and boundary conditions
        if grid.lower_bound[0] != 0.:
            raise ValueError('FBPIC requires the lower radial bound of the '
                'grid (`lower_bound[0]`) to be 0, but it is %s.'
                %grid.lower_bound[0])
        if grid.lower_boundary_conditions[1] != grid.upper_boundary_conditions[1]:
            raise ValueError('FBPIC requires the same boundary condition at '
                'both ends in z, but the lower and upper boundary conditions '
                'of the grid are %r and %r.' %(grid.lower_boundary_conditions[1],
                grid.upper_boundary_conditions[1]))
        # (The boundary conditions are not modified in the PICMI grid itself,
        # so as to leave the input of the user unchanged.)
        boundary_conditions = list( grid.upper_boundary_conditions )
        if grid.lower_boundary_conditions[1] == 'reflective':
            warnings.warn(
            "FBPIC does not support reflective boundary condition in z.\n"
            "The z boundary condition was automatically converted to 'open'.")
            boundary_conditions[1] = 'open'
        if boundary_conditions[1] not in ['periodic', 'open']:
            raise ValueError('FBPIC only supports the boundary conditions '
                "'periodic' and 'open' in z, but the grid has %r."
                %boundary_conditions[1])
        if boundary_conditions[0] not in ['reflective', 'open']:
            raise ValueError('FBPIC only supports the upper radial boundary '
                "conditions 'reflective' and 'open', but the grid has %r."
                %boundary_conditions[0])

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
        self._fbpic.sim = FBPICSimulation(
            Nz=int(grid.number_of_cells[1]), zmin=grid.lower_bound[1], zmax=grid.upper_bound[1],
            Nr=int(grid.number_of_cells[0]), rmax=grid.upper_bound[0], Nm=grid.n_azimuthal_modes,
            dt=dt, use_cuda=True, smoother=smoother, n_order=n_order,
            boundaries={'z':boundary_conditions[1], 'r':boundary_conditions[0]},
            n_guard=n_guard, verbose_level=verbose_level,
            particle_shape=self._get_particle_shape(),
            v_comoving=v_comoving,
            gamma_boost=self.gamma_boost,
            **self._fbpic_arguments(self._simulation_arguments))

        # Set the moving window
        if grid.moving_window_velocity is not None:
            self._fbpic.sim.set_moving_window(grid.moving_window_velocity[-1])


    def _species_particle_shape( self, s ):
        """
        Return the FBPIC particle shape of the PICMI species `s`
        (a species without `particle_shape` uses the one of the Simulation)
        """
        if s.particle_shape is not None:
            return fbpic_particle_shape( s.particle_shape )
        return fbpic_particle_shape( self.particle_shape )


    def _get_particle_shape( self ):
        """
        Return the particle shape of the FBPIC simulation: FBPIC uses the same
        particle shape for all species, which is thus the one of the species
        that were added (or else the one of the Simulation)
        """
        shapes = { self._species_particle_shape( s )
                   for species in self.species
                   for s in species_instances( species ) }
        if len( shapes ) > 1:
            raise ValueError('FBPIC uses the same particle shape for all '
                'species, but the species have the particle shapes %s. (A '
                'species without `particle_shape` uses the one of the '
                'Simulation, which is %r.) To use the same shape for all '
                'species, set it in the Simulation, e.g. with '
                '`Simulation(particle_shape=%r)`, and not in the species.'
                %(sorted(shapes), self.particle_shape,
                  sorted(shapes, key=lambda shape: shape == 'linear')[0]))
        if shapes:
            return shapes.pop()
        return fbpic_particle_shape( self.particle_shape )


    def _fbpic_arguments( self, argnames ):
        """
        Return the FBPIC-specific arguments that the user set, among `argnames`
        (the arguments that are not set keep their default value in FBPIC)
        """
        return { argname: getattr(self, argname) for argname in argnames
                 if getattr(self, argname) is not None }


    def _add_fbpic_laser( self, laser, injection_method ):
        """
        Add a laser to the FBPIC simulation (see `add_laser`)
        """
        # Handle injection method
        if not isinstance(injection_method, PICMI_LaserAntenna):
            raise ValueError('FBPIC only supports a `LaserAntenna` as '
                'injection method of a laser, but got: %s'
                %type(injection_method))
        # Handle laser profile method
        if isinstance(laser, PICMI_GaussianLaser):
            if (laser.propagation_direction[0] != 0.) or \
               (laser.propagation_direction[1] != 0.):
                raise ValueError('FBPIC only supports lasers that propagate '
                    'along z, but the `propagation_direction` of the laser '
                    'is %s.' %laser.propagation_direction)
            # FBPIC lasers propagate either towards positive or negative z
            if laser.propagation_direction[2] > 0:
                propagation_direction = 1
            elif laser.propagation_direction[2] < 0:
                propagation_direction = -1
            else:
                raise ValueError('The `propagation_direction` of the laser '
                                 'cannot be the null vector.')
            if laser.zeta not in [None, 0]:
                raise ValueError('FBPIC does not support a spatial chirp '
                    '(`zeta`) of the laser, but it is %s.' %laser.zeta)
            if laser.beta not in [None, 0]:
                raise ValueError('FBPIC does not support an angular dispersion '
                    '(`beta`) of the laser, but it is %s.' %laser.beta)
            phi2_chirp = laser.phi2
            if phi2_chirp is None:
                phi2_chirp = 0
            cep_phase = laser.phi0
            if cep_phase is None:
                cep_phase = 0
            polarization_angle = np.arctan2(laser.polarization_direction[1],
                                            laser.polarization_direction[0])
            laser_profile = GaussianLaser( a0=laser.a0, waist=laser.waist,
                z0=laser.centroid_position[-1], zf=laser.focal_position[-1],
                tau=laser.duration, theta_pol=polarization_angle,
                lambda0=laser.wavelength, cep_phase=cep_phase,
                phi2_chirp=phi2_chirp,
                propagation_direction=propagation_direction )
        else:
            raise ValueError('Unknown laser profile: %s' %type(laser))

        # Inject the laser
        add_laser_pulse( self._fbpic.sim, laser_profile, method='antenna',
            z0_antenna=injection_method.position[-1],
            gamma_boost=self.gamma_boost )


    def _add_species_generic( self, species, layout, injection_plane_position,
        injection_plane_normal_vector, initialize_self_field ):
        """
        Add a species to the FBPIC simulation
        (see `add_species` and `add_species_through_plane`)
        """

        if isinstance(layout, list):
            raise ValueError('FBPIC does not support more than one layout '
                             'per species.')

        # Loop over species and create FBPIC species
        for s in species_instances( species ):

            # Skip the species whose FBPIC species exists already (e.g. when
            # the creation of a `MultiSpecies` is repeated after an error)
            if s in self._fbpic.species:
                continue

            if isinstance(s.initial_distribution, list):
                raise ValueError('FBPIC does not support more than one '
                                 'initial distribution per species.')

            # FBPIC uses the same particle shape for all species (this can
            # only differ for a species that is added after running)
            if self._species_particle_shape( s ) != \
               self._fbpic.sim.particle_shape:
                raise ValueError('FBPIC uses the same particle shape for all '
                    'species (%r), but species %s has the particle shape %r. '
                    '(A species without `particle_shape` uses the one of the '
                    'Simulation.)' %(self._fbpic.sim.particle_shape,
                    s.name or s.particle_type, self._species_particle_shape(s)))

            # Get their charge and mass
            # (These are not set in the PICMI species itself, so as to leave
            # the input of the user unchanged.)
            charge = s.charge
            mass = s.mass
            if s.particle_type is not None:
                charge = particle_charge[s.particle_type]
                mass = particle_mass[s.particle_type]
            # If `charge_state` is set, redefine the charge and mass
            if s.charge_state is not None:
                charge = s.charge_state*e
                mass -= s.charge_state*m_e

            # Add the species to the FBPIC simulation
            fbpic_species = self._create_new_fbpic_species(s, charge, mass,
                 layout, injection_plane_position,
                 injection_plane_normal_vector, initialize_self_field)

            # Register the FBPIC species that corresponds to the PICMI species
            # (Useful for particle diagnostics and interactions later on)
            self._fbpic.species[s] = fbpic_species


    def _create_new_fbpic_species(self, s, charge, mass, layout,
        injection_plane_position, injection_plane_normal_vector,
        initialize_self_field):

        # Injection plane: FBPIC only supports planes that are perpendicular
        # to z, through which the particles are injected towards positive z.
        # PICMI gives the position of the plane either as a point (whose
        # z coordinate is used) or as a scalar (the z position of the plane),
        # and the normal vector with 2 (r, z) or 3 (x, y, z) components.
        if injection_plane_position is None:
            z_injection_plane = None
        else:
            normal = injection_plane_normal_vector
            if (normal is not None) and ( (len(normal) == 0) or
                any( component != 0 for component in normal[:-1] ) or
                (normal[-1] <= 0) ):
                raise ValueError('FBPIC only supports injection planes that '
                    'are perpendicular to z, with the particles injected '
                    'towards positive z, i.e. an '
                    '`injection_plane_normal_vector` along +z (e.g. '
                    '[0, 0, 1]), but it is %s.' %normal)
            z_injection_plane = float(
                np.atleast_1d( injection_plane_position )[-1] )

        # - For the case of a plasma/beam defined in a gridded layout
        if isinstance(layout, PICMI_GriddedLayout):
            # - Uniform distribution
            if isinstance(s.initial_distribution, PICMI_UniformDistribution):
                n0 = s.initial_distribution.density
                if s.density_scale is not None:
                    n0 *= s.density_scale
                dens_func = None
            # - Analytic distribution
            elif isinstance(s.initial_distribution, PICMI_AnalyticDistribution):
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
            if len(layout.n_macroparticles_per_cell) != 3:
                raise ValueError('FBPIC requires the `n_macroparticles_per_cell` '
                    'of a `GriddedLayout` to have 3 entries (along r, theta and '
                    'z), but it is %s.' %layout.n_macroparticles_per_cell)
            p_nr = layout.n_macroparticles_per_cell[0]
            p_nt = layout.n_macroparticles_per_cell[1]
            p_nz = layout.n_macroparticles_per_cell[2]

            if initialize_self_field or (injection_plane_position is not None):
                if s.initial_distribution.fill_in:
                    raise ValueError('FBPIC does not support `fill_in` for a '
                        'species that is injected through a plane or whose '
                        'self-field is initialized (species %s).'
                        %(s.name or s.particle_type))

                gamma0_beta0 = s.initial_distribution.directed_velocity[-1]/c
                gamma0 = ( 1 + gamma0_beta0**2 )**.5
                dist = s.initial_distribution
                fbpic_species = add_particle_bunch( self._fbpic.sim,
                    q=charge, m=mass, gamma0=gamma0, n=n0,
                    dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                    p_zmin=dist.lower_bound[-1] if dist.lower_bound[-1] is not None else -np.inf,
                    p_zmax=dist.upper_bound[-1] if dist.upper_bound[-1] is not None else +np.inf,
                    p_rmin=0,
                    p_rmax=dist.upper_bound[0] if dist.upper_bound[0] is not None else +np.inf,
                    boost=self._fbpic.sim.boost,
                    z_injection_plane=z_injection_plane,
                    initialize_self_field=initialize_self_field,
                    boost_positions_in_dens_func=True )
            else:
                dist = s.initial_distribution
                fbpic_species = self._fbpic.sim.add_new_species(
                    q=charge, m=mass, n=n0,
                    dens_func=dens_func, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                    p_zmin=dist.lower_bound[-1] if dist.lower_bound[-1] is not None else -np.inf,
                    p_zmax=dist.upper_bound[-1] if dist.upper_bound[-1] is not None else +np.inf,
                    p_rmax=dist.upper_bound[0] if dist.upper_bound[0] is not None else +np.inf,
                    continuous_injection=s.initial_distribution.fill_in,
                    boost_positions_in_dens_func=True )

        # - For the case of a Gaussian beam
        elif isinstance(s.initial_distribution, PICMI_GaussianBunchDistribution) \
             and isinstance(layout, PICMI_PseudoRandomLayout):
            if layout.n_macroparticles is None:
                raise ValueError('For a Gaussian bunch, FBPIC requires the '
                    '`PseudoRandomLayout` to be defined with '
                    '`n_macroparticles` (instead of `n_macroparticles_per_cell`).')
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
            fbpic_species = add_particle_bunch_gaussian( self._fbpic.sim,
                                q=charge, m=mass,
                                gamma0=gamma0, sig_gamma=sig_gamma,
                                sig_r=sig_r0, sig_z=sig_z, n_emit=n_emit,
                                n_physical_particles=n_physical_particles,
                                n_macroparticles=layout.n_macroparticles,
                                zf=zf, tf=tf, boost=self._fbpic.sim.boost,
                                z_injection_plane=z_injection_plane,
                                initialize_self_field=initialize_self_field )

        # - For the case of an empty species
        elif (s.initial_distribution is None) and (layout is None):
            fbpic_species = self._fbpic.sim.add_new_species(q=charge, m=mass)

        else:
            raise ValueError('Unknown combination of layout and distribution')

        return fbpic_species


    def _pending_interactions( self ):
        """
        Return the interactions (of the simulation itself and of its species)
        that have not been set up in the FBPIC simulation yet
        """
        interactions = list( self.interactions )
        for species in self.species:
            for s in species_instances( species ):
                interactions += s.interactions
        # An interaction can be listed several times (e.g. in the simulation
        # and in a species), but is set up only once
        pending = []
        for interaction in interactions:
            if (interaction not in self._fbpic.interactions) and \
               (interaction not in pending):
                pending.append( interaction )
        return pending


    def _setup_interactions( self ):
        """
        Set up the interactions for which all the species involved have been
        added to the FBPIC simulation already.

        (An interaction whose species are not added yet is set up later on,
        and reported by `step` if its species are never added.)
        """
        for interaction in self._pending_interactions():
            if not isinstance( interaction, PICMI_FieldIonization ):
                raise ValueError(
                    'Unknown interaction: %s' %type(interaction))
            if interaction.model != 'ADK':
                raise ValueError("FBPIC only supports the 'ADK' model "
                    "for field ionization.")
            picmi_source = interaction.ionized_species
            fbpic_source = self._find_fbpic_species( picmi_source )
            fbpic_target = self._find_fbpic_species( interaction.product_species )
            if (fbpic_source is None) or (fbpic_target is None):
                # Wait until both species have been added to the simulation
                continue
            # PICMI defines `charge_state` as a float, while FBPIC expects an
            # integer ionization level (a float would turn the array of
            # ionization levels into an array of floats)
            level_start = picmi_source.charge_state
            if level_start is None:
                level_start = 0
            fbpic_source.make_ionizable(
                element=picmi_source.particle_type,
                level_start=int(level_start),
                target_species=fbpic_target )
            self._fbpic.interactions.append( interaction )


    def _find_fbpic_species( self, s ):
        """
        Return the FBPIC species that corresponds to the PICMI species `s`,
        or None if `s` was not added to the FBPIC simulation (yet)

        The PICMI species are identified by object and otherwise by their name,
        if it is unique: e.g. `picmistandard.load` does not preserve shared
        objects, so that the species of a diagnostic become copies of the
        species of the simulation, which are thus found by their name.
        """
        if s in self._fbpic.species:
            return self._fbpic.species[ s ]
        if s.name is not None:
            matches = [ fbpic_species for picmi_species, fbpic_species
                        in self._fbpic.species.items()
                        if picmi_species.name == s.name ]
            if len( matches ) == 1:
                return matches[0]
        return None


    def _get_fbpic_species( self, s ):
        """
        Return the FBPIC species that corresponds to the PICMI species `s`
        """
        fbpic_species = self._find_fbpic_species( s )
        if fbpic_species is None:
            raise ValueError('The species %s needs to be added to the '
                'simulation (with `add_species`), before it can be used. '
                '(Species are identified by object, or else by their name '
                'if it is unique.)' %(s.name or s.particle_type))
        return fbpic_species


    def _get_species_dict( self, diagnostic ):
        """
        Return the FBPIC species of `diagnostic`, in a dictionary
        that is labeled by the name of the PICMI species
        """
        if diagnostic.species is None:
            # When no species is specified, all species are written
            picmi_species = self.species
        elif isinstance( diagnostic.species, list ):
            picmi_species = diagnostic.species
        else:
            picmi_species = [ diagnostic.species ]

        species_dict = {}
        for species in picmi_species:
            for s in species_instances( species ):
                if s.name is None:
                    raise ValueError('When using a species in a diagnostic, '
                                      'its name must be set.')
                species_dict[s.name] = self._get_fbpic_species( s )
        return species_dict


    def _add_fbpic_diagnostic( self, diagnostic ):
        """
        Add a diagnostic to the FBPIC simulation (see `add_diagnostic`)
        """
        # The FBPIC diagnostics of this PICMI diagnostic
        new_diags = []

        # Handle iteration_min/max in regular diagnostic
        if isinstance(diagnostic, (PICMI_FieldDiagnostic, PICMI_ParticleDiagnostic)):
            if diagnostic.step_min is None:
                iteration_min = 0
            else:
                iteration_min = diagnostic.step_min
            if diagnostic.step_max is None:
                iteration_max = np.inf
            else:
                iteration_max = diagnostic.step_max

        # Register field diagnostic
        if isinstance(diagnostic,
                      (PICMI_FieldDiagnostic, PICMI_LabFrameFieldDiagnostic)):
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

        if isinstance(diagnostic, PICMI_FieldDiagnostic):

            diag = FieldDiagnostic(
                    period=diagnostic.period,
                    fldobject=self._fbpic.sim.fld,
                    comm=self._fbpic.sim.comm,
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
                    for species in self.species:
                        for s in species_instances( species ):
                            if s.name == sname:
                                species_dict[s.name] = \
                                    self._get_fbpic_species( s )
                pdd_diag = ParticleChargeDensityDiagnostic(
                            period=diagnostic.period,
                            sim=self._fbpic.sim,
                            species=species_dict,
                            write_dir=diagnostic.write_dir,
                            iteration_min=iteration_min,
                            iteration_max=iteration_max)
                new_diags.append( pdd_diag )

        elif isinstance(diagnostic, PICMI_LabFrameFieldDiagnostic):
            diag = BackTransformedFieldDiagnostic(
                    zmin_lab=diagnostic.grid.lower_bound[1],
                    zmax_lab=diagnostic.grid.upper_bound[1],
                    v_lab=c,
                    dt_snapshots_lab=diagnostic.dt_snapshots,
                    Ntot_snapshots_lab=diagnostic.num_snapshots,
                    gamma_boost=self.gamma_boost,
                    period=100,
                    fldobject=self._fbpic.sim.fld,
                    comm=self._fbpic.sim.comm,
                    fieldtypes=diagnostic.data_list,
                    write_dir=diagnostic.write_dir)
        # Register particle diagnostic
        elif isinstance(diagnostic, (PICMI_ParticleDiagnostic,
                                     PICMI_LabFrameParticleDiagnostic)):
            species_dict = self._get_species_dict( diagnostic )
            if diagnostic.data_list is None:
                data_list = ['position', 'momentum', 'weighting']
            else:
                data_list = diagnostic.data_list
            if isinstance(diagnostic, PICMI_ParticleDiagnostic):
                diag = ParticleDiagnostic(
                    period=diagnostic.period,
                    species=species_dict,
                    comm=self._fbpic.sim.comm,
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
                    fldobject=self._fbpic.sim.fld,
                    species=species_dict,
                    comm=self._fbpic.sim.comm,
                    particle_data=data_list,
                    write_dir=diagnostic.write_dir)
        else:
            raise ValueError("Unrecognized `diagnostic` type.")

        # Add the diagnostics to the FBPIC simulation (at once, so that none
        # of them is added if the creation of another one fails)
        new_diags.append( diag )
        self._fbpic.sim.diags.extend( new_diags )

    def _add_fbpic_applied_field( self, applied_field ):
        """
        Add an applied field to the FBPIC simulation (see `add_applied_field`)
        """
        # The FBPIC external fields of this PICMI applied field
        external_fields = []

        if isinstance(applied_field, PICMI_Mirror):
            if applied_field.z_front_location is None:
                raise ValueError('FBPIC only supports mirrors that are '
                    'perpendicular to z, i.e. with a `z_front_location`.')
            mirror = Mirror( z_lab=applied_field.z_front_location,
                             n_cells=applied_field.number_of_cells,
                             gamma_boost=self._fbpic.sim.boost.gamma0 )
            self._fbpic.sim.mirrors.append( mirror )

        elif isinstance(applied_field, PICMI_ConstantAppliedField):
            # TODO: Handle bounds
            for field_name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                field_value = getattr( applied_field, field_name )
                if field_value is None:
                    continue
                # Pass it to FBPIC
                external_fields.append(
                    ExternalField( constant_field_func(field_value),
                                   field_name, 1., 0.)
                )

        elif isinstance(applied_field, PICMI_AnalyticAppliedField):
            # TODO: Handle bounds
            for field_name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                # Extract expression and execute it inside a function definition
                expression = getattr( applied_field, field_name+'_expression' )
                if expression is None:
                    continue
                define_function_code = \
                """def fieldfunc( F, x, y, z, t, amplitude, length_scale ):\n    return( F + amplitude * ( %s ) )""" %expression
                # Define the function in a dedicated namespace, which
                # contains the functions and constants of the `math` module
                # (e.g. sin, exp, pi) and the user-defined variables
                namespace = { k: v for k, v in vars(math).items()
                              if not k.startswith('_') }
                namespace.update( applied_field.user_defined_kw )
                exec( define_function_code, namespace )
                fieldfunc = namespace['fieldfunc']
                # Pass it to FBPIC
                external_fields.append(
                    ExternalField( fieldfunc, field_name, 1., 0.)
                )

        else:
            raise ValueError("Unrecognized `applied_field` type.")

        # Add the external fields to the FBPIC simulation (at once, so that
        # none of them is added if the creation of another one fails)
        self._fbpic.sim.external_fields.extend( external_fields )


    # Redefine the method `step` of the parent class
    def step(self, nsteps=None):
        if nsteps is None:
            nsteps = self.max_steps
        # Create the FBPIC objects of the PICMI input that was given so far
        self._create_fbpic_objects()
        # Check that no interaction is left out, e.g. because one of its
        # species was never added to the simulation
        pending_interactions = self._pending_interactions()
        if pending_interactions:
            raise ValueError('The species of the following interactions were '
                'not added to the simulation: %s'
                %', '.join([type(interaction).__name__
                            for interaction in pending_interactions]))
        self._fbpic.sim.step( nsteps , **self._fbpic_arguments(self._step_arguments))
