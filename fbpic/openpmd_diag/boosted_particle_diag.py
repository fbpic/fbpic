# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class BoostedParticleDiagnostic

Major features:
- The class reuses the existing methods of ParticleDiagnostic
  as much as possible through class inheritance
- The class implements memory buffering of the slices, so as
  not to write to disk at every timestep
"""
import os
import math
import numpy as np
from scipy.constants import c, e
from .particle_diag import ParticleDiagnostic

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    from .cuda_methods import extract_slice_from_gpu

class BackTransformedParticleDiagnostic(ParticleDiagnostic):
    """
    Class that writes the particles *in the lab frame*,
    from a simulation in the boosted frame

    Particles are extracted from the simulation in slices each time step
    and buffered in memory before writing to disk. On the CPU, slices of
    particles are directly selected from the particle arrays of the species.
    On the GPU, first particles within an area of cells surrounding the
    output planes are extracted from the GPU particle arrays and stored in
    a smaller GPU array, which is then copied to the CPU for selection.
    The mechanism of extracting the particles within the outputplane-area
    on the GPU relies on particle arrays being sorted on the GPU. For the
    back-transformation to the Lab frame, interpolation in space is applied,
    but no interpolation for the particle velocities is applied.
    """
    def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshots_lab,
                 Ntot_snapshots_lab, gamma_boost, period, fldobject,
                 particle_data=["position", "momentum", "weighting"],
                 select=None, write_dir=None, species={"electrons": None},
                 comm = None):
        """
        Initialize diagnostics that retrieve the data in the lab frame,
        as a series of snapshot (one file per snapshot),
        within a virtual moving window defined by zmin_lab, zmax_lab, v_lab.

        The parameters defined below are specific to the back-transformed
        diagnostics. See the documentation of `ParticleDiagnostic` for
        the other parameters.

        .. warning::

            The output of the gathered fields on the particles
            (``particle_data=["E", "B"]``) and of the Lorentz factor
            (``particle_data=["gamma"]``) is not currently supported
            for ``BackTransformedParticleDiagnostic``.

        Parameters
        ----------
        zmin_lab: float (in meters)
            The position of the left edge of the virtual moving window,
            *in the lab frame*, at t=0
        zmax_lab: float (in meters)
            The position of the right edge of the virtual moving window,
            *in the lab frame*, at t=0

        v_lab: float (m.s^-1)
            Speed of the moving window *in the lab frame*

        dt_snapshots_lab: float (seconds)
            Time interval *in the lab frame* between two successive snapshots

        Ntot_snapshots_lab: int
            Total number of snapshots that this diagnostic will produce

        period: int
            Number of iterations for which the data is accumulated in memory,
            before finally writing it to the disk.

        fldobject : a Fields object,
            The Fields object of the simulation, that is needed to
            extract some information about the grid
        """
        # Do not leave write_dir as None, as this may conflict with
        # the default directory ('./diags') in which diagnostics in the
        # boosted frame are written
        if write_dir is None:
            write_dir = 'lab_diags'

        # Initialize Particle diagnostic normal attributes
        ParticleDiagnostic.__init__(self, period, species,
            comm, particle_data, select, write_dir)

        # Register the Field object
        self.fld = fldobject

        # Register the boost quantities
        self.gamma_boost = gamma_boost
        self.inv_gamma_boost = 1./gamma_boost
        self.beta_boost = np.sqrt(1. - self.inv_gamma_boost**2)
        self.inv_beta_boost = 1./self.beta_boost

        # Create the list of LabSnapshot objects
        self.snapshots = []
        for i in range( Ntot_snapshots_lab ):
            t_lab = i*dt_snapshots_lab
            snapshot = LabSnapshot( t_lab,
                                    zmin_lab + v_lab*t_lab,
                                    zmax_lab + v_lab*t_lab,
                                    self.dt,
                                    self.write_dir, i ,self.species_dict )
            self.snapshots.append(snapshot)
            # Initialize a corresponding empty file to store particles
            self.create_file_empty_slice(
                    snapshot.filename, i, snapshot.t_lab, self.dt)

        # Create the ParticleCatcher object
        # (This object will extract the particles (slices) that crossed the
        # output plane at each iteration.)
        self.particle_catcher = ParticleCatcher(
            self.gamma_boost, self.beta_boost, self.fld )

    def write( self, iteration ):
        """
        Redefines the method write of the parent class ParticleDiagnostic

        Parameters
        ----------
        iteration : int
            Current iteration of the boosted frame simulation
        """
        # At each timestep, store a slice of the particles in memory buffers
        self.store_snapshot_slices(iteration)

        # Every self.period, write the buffered slices to disk
        if iteration % self.period == 0:
            self.flush_to_disk()

    def store_snapshot_slices( self, iteration ):
        """
        Store slices of the particles in the memory buffers of the
        corresponding lab snapshots

        Parameters
        ----------
        iteration : int
            Current iteration of the boosted frame simulation
        """
        # Find the limits of the local subdomain at this iteration
        zmin_boost = self.fld.interp[0].zmin
        zmax_boost = self.fld.interp[0].zmax

        # Extract the current time in the boosted frame
        time = iteration * self.dt

        # Loop through the labsnapshots
        for snapshot in self.snapshots:

            # Update the positions of the output slice of this snapshot
            # in the lab and boosted frame (current_z_lab and current_z_boost)
            snapshot.update_current_output_positions( time,
                self.inv_gamma_boost, self.inv_beta_boost)

            # For this snapshot:
            # - check if the output position *in the boosted frame*
            #   is in the current local domain
            # - check if the output position *in the lab frame*
            #   is within the lab-frame boundaries of the current snapshot
            if ( (snapshot.current_z_boost >= zmin_boost) and \
                 (snapshot.current_z_boost < zmax_boost) and \
                 (snapshot.current_z_lab >= snapshot.zmin_lab) and \
                 (snapshot.current_z_lab < snapshot.zmax_lab) ):

                # Loop through the particle species and register the
                # data dictionaries in the snapshot objects (buffering)
                for species_name in self.species_names_list:
                    species = self.species_dict[species_name]
                    # Extract the slice of particles
                    slice_data_dict = self.particle_catcher.extract_slice(
                        species, snapshot.current_z_boost,
                        snapshot.prev_z_boost, time, self.select)
                    # Register new slice in the LabSnapshot
                    snapshot.register_slice( slice_data_dict, species_name )

    def flush_to_disk(self):
        """
        Writes the buffered slices of particles to the disk. Erase the
        buffered slices of the LabSnapshot objects
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in self.snapshots:

            # Compact the successive slices that have been buffered
            # over time into a single array
            for species_name in self.species_names_list:

                # Get list of quantities to be written to file
                quantities_in_file = self.array_quantities_dict[species_name]

                # Compact the slices in a single array (on each proc)
                local_particle_dict = snapshot.compact_slices(species_name,
                                    quantities_in_file )

                # Gather the slices on the first proc
                if self.comm is not None and self.comm.size > 1:
                    particle_dict = self.gather_particle_arrays(
                        local_particle_dict, quantities_in_file )
                else:
                    particle_dict = local_particle_dict

                # The first proc writes this array to disk
                # (if this snapshot has new slices)
                if self.rank==0:
                    self.write_slices( particle_dict, species_name, snapshot )

                # Erase the previous slices
                snapshot.buffered_slices[species_name] = []

    def gather_particle_arrays( self, local_dict, quantities_in_file ):
        """
        Gather the compacted arrays of particle slices, on the proc `root`

        Parameters:
        -----------
        local_dict: A dictionary of 1d arrays of shape (n_particles_local,)
            A dictionary that contains the quantities on one MPI rank.
        quantities_in_file: list of strings
            The quantities that will be written into the openPMD
            file, for this species.

        Returns:
        --------
        gathered_dict: A dictionary of 1d arrays of shape (n_particles_total,)
        (None is returned on all other processors than root.)
        """
        # Send the local number of particles to all procs
        n_particles_local = len( local_dict[ quantities_in_file[0] ] )
        n_particles_list = self.comm.mpi_comm.allgather( n_particles_local )

        # Prepare the send and receive buffers
        gathered_dict = {}
        n_particles_tot = sum( n_particles_list )
        # Loop through the quantities and perform the MPI gather
        for quantity in quantities_in_file:
            gathered_dict[quantity] = self.comm.gather_ptcl_array(
                local_dict[quantity], n_particles_list, n_particles_tot )

        # Return the gathered dictionary
        return( gathered_dict )

    def write_slices( self, particle_dict, species_name, snapshot ):
        """
        For one given snapshot, write the slices of the
        different species to an openPMD file

        Parameters
        ----------
        particle_dict: A dictionary of 1d arrays of shape (n_particles_local,)
            A dictionary that contains the different particle quantities,
            whose keys are self.arrays_quantities[species_name]

        species_name: String
            A String that acts as the key for the buffered_slices dictionary

        snapshot: a LabSnaphot object
        """
        # Open the file without parallel I/O in this implementation
        f = self.open_file(snapshot.filename)
        particle_path = "/data/%d/particles/%s" %(snapshot.iteration,
            species_name)
        species_grp = f[particle_path]

        # Loop over the different quantities that should be written
        for quantity in self.array_quantities_dict[species_name]:

            if quantity in ["x","y","z"]:
                path = "position/%s" %(quantity)
                data = particle_dict[ quantity ]
                self.write_particle_slices(species_grp, path, data, quantity)

            elif quantity in ["ux","uy","uz"]:
                path = "momentum/%s" %(quantity[-1])
                data = particle_dict[ quantity ]
                self.write_particle_slices( species_grp, path, data, quantity)

            elif quantity in ["w", "charge", "id"]:
                if quantity == "w":
                    path = "weighting"
                else:
                    path = quantity
                data = particle_dict[ quantity ]
                self.write_particle_slices(species_grp, path, data, quantity)

        # Close the file
        f.close()

    def write_particle_slices( self, species_grp, path, data, quantity ):
        """
        Writes each quantity of the buffered dataset to the disk, the
        final step of the writing
        """
        dset = species_grp[path]
        index = dset.shape[0]

        # Resize the h5py dataset
        dset.resize(index+len(data), axis=0)

        # Write the data to the dataset at correct indices
        dset[index:] = data

    def create_file_empty_slice( self, fullpath, iteration, time, dt ):
        """
        Create an openPMD file with empty meshes and setup all its attributes

        Parameters
        ----------
        fullpath: string
            The absolute path to the file to be created

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this ibteration

        dt: float (seconds)
            The timestep of the simulation
        """
        # Create the file
        f = self.open_file( fullpath )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, dt )
            # Setup the meshes group (contains all the particles)
            particle_path = "/data/%d/particles/" %iteration

            for species_name in self.species_names_list:
                species = self.species_dict[species_name]
                species_path = particle_path+"%s/" %(species_name)
                # Create and setup the h5py.Group species_grp
                species_grp = f.require_group( species_path )
                self.setup_openpmd_species_group( species_grp, species,
                            self.constant_quantities_dict[species_name])

                # Loop over the different quantities that should be written
                # and setup the corresponding datasets
                for quantity in self.array_quantities_dict[species_name]:

                    if quantity in ["x", "y", "z"]:
                        quantity_path = "position/%s" %(quantity)
                        dset = species_grp.require_dataset(
                                quantity_path, (0,),
                                maxshape=(None,), dtype='f8')
                        self.setup_openpmd_species_component( dset, quantity )

                    elif quantity in ["ux", "uy", "uz"]:
                        quantity_path = "momentum/%s" %(quantity[-1])
                        dset = species_grp.require_dataset(
                                quantity_path, (0,),
                                maxshape=(None,), dtype='f8')
                        self.setup_openpmd_species_component( dset, quantity )

                    elif quantity in ["w", "id", "charge"]:
                        if quantity == "w":
                            particle_var = "weighting"
                        else:
                            particle_var = quantity
                        if quantity == "id":
                            dtype = 'uint64'
                        else:
                            dtype = 'f8'
                        dset = species_grp.require_dataset(
                            particle_var, (0,), maxshape=(None,), dtype=dtype )
                        self.setup_openpmd_species_component( dset, quantity )
                        self.setup_openpmd_species_record(
                            species_grp[particle_var], particle_var )

                    else :
                        raise ValueError(
                            "Invalid quantity for particle output: %s"
                            %(quantity) )

                # Setup the hdf5 groups for "position" and "momentum"
                if self.rank == 0:
                    if "x" in self.array_quantities_dict[species_name]:
                        self.setup_openpmd_species_record(
                            species_grp["position"], "position" )
                    if "ux" in self.array_quantities_dict[species_name]:
                        self.setup_openpmd_species_record(
                            species_grp["momentum"], "momentum" )

            # Close the file
            f.close()

class LabSnapshot:
    """
    Class that stores data relative to one given snapshot
    in the lab frame (i.e. one given *time* in the lab frame)
    """
    def __init__( self, t_lab, zmin_lab, zmax_lab, dt,
                  write_dir, i, species_dict ):
        """
        Initialize a LabSnapshot

        Parameters
        ----------
        t_lab: float (seconds)
            Time of this snapshot *in the lab frame*

        zmin_lab, zmax_lab: floats (meters)
            Longitudinal limits of this snapshot

        write_dir: string
            Absolute path to the directory where the data for
            this snapshot is to be written

        dt : float (s)
            The timestep of the simulation in the boosted frame

        i: int
            Number of the file where this snapshot is to be written

        species_dict: dict
            Contains all the species name of the species object
            (inherited from Warp)
        """
        # Deduce the name of the filename where this snapshot writes
        self.filename = os.path.join( write_dir, 'hdf5/data%08d.h5' %i)
        self.iteration = i
        self.dt = dt

        # Time and boundaries in the lab frame (constant quantities)
        self.zmin_lab = zmin_lab
        self.zmax_lab = zmax_lab
        self.t_lab = t_lab

        # Positions where the fields are to be registered
        # (Change at every iteration)
        self.current_z_lab = 0
        self.current_z_boost = 0

        # Initialize empty dictionary to buffer the slices for each species
        self.buffered_slices = {}
        for species in species_dict:
            self.buffered_slices[species] = []

    def update_current_output_positions( self, t_boost, inv_gamma, inv_beta ):
        """
        Update the current and previous positions of output for this snapshot,
        so that it corresponds to the time t_boost in the boosted frame

        Parameters
        ----------
        t_boost: float (seconds)
            Time of the current iteration, in the boosted frame

        inv_gamma, inv_beta: floats
            Inverse of the Lorentz factor of the boost, and inverse
            of the corresponding beta
        """
        # Some shorcuts for further calculation's purposes
        t_lab = self.t_lab
        t_boost_prev = t_boost - self.dt

        # This implements the Lorentz transformation formulas,
        # for a snapshot having a fixed t_lab
        self.current_z_boost = (t_lab*inv_gamma - t_boost)*c*inv_beta
        self.prev_z_boost = (t_lab*inv_gamma - t_boost_prev)*c*inv_beta
        self.current_z_lab = (t_lab - t_boost*inv_gamma)*c*inv_beta
        self.prev_z_lab = (t_lab - t_boost_prev*inv_gamma)*c*inv_beta

    def register_slice( self, slice_data_dict, species ):
        """
        Store the slice of particles represented by slice_data_dict

        Parameters
        ----------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation, including optional integer arrays (e.g. "id"),
            as given by the ParticleCatcher object

        species: String, key of the species_dict
            Act as the key for the buffered_slices dictionary
        """
        # Store the values
        self.buffered_slices[species].append(slice_data_dict)

    def compact_slices( self, species, quantities_in_file ):
        """
        Compact the successive slices that have been buffered
        over time into a single array.

        Parameters
        ----------
        species: String, key of the species_dict
            Act as the key for the buffered_slices dictionary

        quantities_in_file: list of strings
            The quantities that will be written into the openPMD
            file, for this species.

        Returns
        -------
        particle_data_dict: dictionary of 1D float and integer arrays
            A dictionary that contains only the particle quantities
            that will be finally written to file, with compacted arrays.
        """
        # Prepare dictionary
        particle_data_dict = {}

        # Loop through the particle quantities that will be written to file,
        # and compact the buffered arrays into a single array
        if self.buffered_slices[species] != []:
            for quantity in quantities_in_file:
                buffered_arrays = [ slice_dict[quantity] \
                            for slice_dict in self.buffered_slices[species] ]
                particle_data_dict[quantity] = np.concatenate(buffered_arrays)
        else:
            for quantity in quantities_in_file:
                if quantity == 'id':
                    dtype = np.uint64
                else:
                    dtype = np.float64
                particle_data_dict[quantity] = np.zeros( (0,), dtype=dtype )

        return(particle_data_dict)

class ParticleCatcher:
    """
    Class that extracts, Lorentz-transforms and gathers particles
    """
    def __init__( self, gamma_boost, beta_boost, fldobject ):
        """
        Initialize the ParticleCatcher object

        Parameters
        ----------
        gamma_boost, beta_boost: float
            The Lorentz factor of the boost and the corresponding beta

        fldobject : a Fields object,
            The Fields object of the simulation, that is needed to
            extract some information about the grid
        """
        # Some attributes necessary for particle selections
        self.gamma_boost = gamma_boost
        self.beta_boost = beta_boost

        # Register the fields object
        self.fld = fldobject
        self.dt = self.fld.dt

    def extract_slice( self, species, current_z_boost, previous_z_boost,
                       t, select=None ):
        """
        Extract a slice of the particles at z_boost and if select is present,
        extract only the particles that satisfy the given criteria

        Parameters
        ----------
        species : A ParticleObject
            Contains the particle attributes to output

        current_z_boost, previous_z_boost : float (m)
            Current and previous position of the output plane
            in the boosted frame

        t : float (s)
            Current time of the simulation in the boosted frame

        select : dict
            A set of rules defined by the users in selecting the particles
            z: {"uz" : [50, 100]} for particles which have normalized
            momenta between 50 and 100

        Returns
        -------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")
        """
        # Get a dictionary containing the particle data
        # When running on the GPU, this only copies to CPU the particles
        # within a small area around the output plane.
        # (Return the result in the form of a dictionary of 1darrays)
        particle_data_dict = self.get_particle_data(
                species, current_z_boost, previous_z_boost, t )

        # Get the selection of particles (slice) that crossed the
        # output plane during the last iteration
        # (Return the result in the form of a dictionary of smaller 1darrays)
        slice_data_dict = self.get_particle_slice(
                particle_data_dict, current_z_boost, previous_z_boost )

        # Backpropagate particles to correct output position and
        # transform particle attributes to the lab frame
        # (Modifies the arrays of `slice_data_dict` in place.)
        slice_data_dict = self.interpolate_particles_to_lab_frame(
            slice_data_dict, current_z_boost, t )

        # Choose the particles based on the select criteria defined by the
        # users. Notice: this implementation still comes with a cost,
        # one way to optimize it would be to do the selection before Lorentz
        # transformation back to the lab frame
        if (select is not None):
            # Find the particles that should be selected and resize
            # the arrays in `slice_data_dict` accordingly
            slice_data_dict = self.apply_selection(select, slice_data_dict)

        # Convert data to the OpenPMD standard
        slice_data_dict = self.apply_opmd_standard( slice_data_dict, species )

        return slice_data_dict

    def get_particle_data( self, species, current_z_boost,
                           previous_z_boost, t ):
        """
        Extract the particle data from the species object.
        In case CUDA is used, only a selection of particles
        (i.e. particles that are within cells corresponding
        to the immediate neighborhood of the output plane)
        is received from the GPU (increases performance).

        Parameters
        ----------
        species : A ParticleObject
            Contains the particle attributes to output
        current_z_boost, previous_z_boost : float (m)
            Current and previous position of the output plane
            in the boosted frame
        t : float (s)
            Current time of the simulation in the boosted frame

        Returns
        -------
        particle_data : A dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")
        """
        # CPU
        if species.use_cuda is False:
            # Create a dictionary containing the particle attributes
            particle_data = {
                'x': species.x, 'y': species.y, 'z': species.z,
                'ux': species.ux, 'uy' : species.uy, 'uz': species.uz,
                'w': species.w, 'inv_gamma': species.inv_gamma }
            # Optional integer quantities
            if species.ionizer is not None:
                particle_data['charge'] = species.ionizer.ionization_level
            if species.tracker is not None:
                particle_data['id'] = species.tracker.id
        # GPU
        else:
            # Check if particles are sorted, otherwise sort them
            if species.sorted == False:
                species.sort_particles(fld=self.fld)
                # The particles are now sorted and rearranged
                species.sorted = True
            # Precalculating quantities and shortcuts
            dt = self.fld.dt
            dz = self.fld.interp[0].dz
            zmin = self.fld.interp[0].zmin
            pref_sum = species.prefix_sum
            pref_sum_shift = species.prefix_sum_shift
            Nz, Nr = species.grid_shape
            # Calculate cell area to get particles from
            # - Get z indices of the slices in which to get the particles
            # (mirrors the index calculation in `get_cell_idx_per_particle`)
            iz_curr = int(math.ceil((current_z_boost-zmin-0.5*dz)/dz))
            iz_prev = int(math.ceil((previous_z_boost-zmin-0.5*dz + dt*c)/dz)) + 1
            # - Get the prefix sum values that correspond to these indices
            #   (Take into account potential shift due to the moving window)
            z_cell_curr = iz_curr + pref_sum_shift
            if z_cell_curr <= 0:
                pref_sum_curr = 0
            elif z_cell_curr > Nz:
                pref_sum_curr = species.Ntot
            else:
                pref_sum_curr = int(pref_sum[ z_cell_curr*(Nr+1) - 1 ])
            z_cell_prev = iz_prev + pref_sum_shift
            if z_cell_prev <= 0:
                pref_sum_prev = 0
            elif z_cell_prev > Nz:
                pref_sum_prev = species.Ntot
            else:
                pref_sum_prev = int(pref_sum[ z_cell_prev*(Nr+1) - 1 ])
            # Calculate number of particles in this area (N_area)
            N_area = pref_sum_prev - pref_sum_curr
            # Check if there are particles to extract
            if N_area > 0:
                # Only copy a particle slice of size N_area from the GPU
                particle_data = extract_slice_from_gpu(
                                    pref_sum_curr, N_area, species )
            else:
                # Empty particle data
                particle_data = {}
                for var in ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'inv_gamma']:
                    particle_data[var] = np.empty( (0,), dtype=np.float64 )
                # Empty optional integer quantities
                if species.ionizer is not None:
                    particle_data['charge'] = np.empty( (0,), dtype=np.uint64 )
                if species.tracker is not None:
                    particle_data['id'] = np.empty( (0,), dtype=np.uint64 )

        return( particle_data )

    def get_particle_slice( self, particle_data, current_z_boost,
                             previous_z_boost ):
        """
        Get the selection of particles that crossed the output
        plane during the last iteration.

        Parameters
        ----------
        particle_data : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")

        current_z_boost, previous_z_boost : float (m)
            Current and previous position of the output plane
            in the boosted frame

        Returns
        -------
        slice_data : dictionary of 1D float and integer arrays
            Contains the same keys as particle_data, but with smaller
            arrays which contain only the particles of the slice.
        """
        # Shortcut
        pd = particle_data

        # Calculate current and previous position in z
        current_z = pd['z']
        previous_z = pd['z']-pd['uz']*pd['inv_gamma']*c*self.dt

        # A particle array for mapping purposes
        particle_indices = np.arange(len(current_z))

        # For this snapshot:
        # - check if the output position *in the boosted frame*
        #   crosses the zboost in a forward motion
        # - check if the output position *in the boosted frame*
        #   crosses the zboost_prev in a backward motion
        selected_indices = np.compress((
            ((current_z >= current_z_boost)&(previous_z <= previous_z_boost))|
            ((current_z <= current_z_boost)&(previous_z >= previous_z_boost))),
            particle_indices)

        # Create dictionary which contains only the selected particles
        slice_data = {}
        for quantity in pd.keys():
            slice_data[quantity] = \
                np.take( particle_data[quantity], selected_indices)

        return( slice_data )

    def interpolate_particles_to_lab_frame( self, slice_data_dict,
                                                    current_z_boost, t ):
        """
        Transform the particle quantities from the boosted frame to the
        lab frame. These are classical Lorentz transformation equations.

        `slice_data_dict` is modified in place.

        Parameters
        ----------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")

        current_z_boost : float (m)
            Current position of the output plane in the boosted frame

        t : float (s)
            Current time of the simulation in the boosted frame

        Return
        ------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")
        """
        # Shortcuts for particle attributes
        x = slice_data_dict['x']
        y = slice_data_dict['y']
        z = slice_data_dict['z']
        ux = slice_data_dict['ux']
        uy = slice_data_dict['uy']
        uz = slice_data_dict['uz']
        inv_gamma = slice_data_dict['inv_gamma']

        # Calculate time (t_cross) when particle and plane intersect
        # Velocity of the particles
        v_z = uz*inv_gamma*c
        # Velocity of the plane
        v_plane = -c/self.beta_boost
        # Time in the boosted frame when particles cross the output plane
        t_cross = t - (current_z_boost - z) / (v_plane - v_z)

        # Push particles to position of plane intersection
        x += c*(t_cross - t)*inv_gamma*ux
        y += c*(t_cross - t)*inv_gamma*uy
        z += c*(t_cross - t)*inv_gamma*uz

        # Back-transformation of position with updated time (t_cross)
        z_lab = self.gamma_boost*( z + self.beta_boost*c*t_cross )

        # Back-transformation of momentum
        gamma = 1./inv_gamma
        uz_lab = self.gamma_boost*uz + gamma*(self.beta_boost*self.gamma_boost)

        # Replace the arrays that have been modified, in `slice_data_dict`
        slice_data_dict['x'] = x
        slice_data_dict['y'] = y
        slice_data_dict['z'] = z_lab
        slice_data_dict['uz'] = uz_lab
        # Remove `inv_gamma`, since it is not needed anymore, now at the
        # Lorentz transform has been performed.
        slice_data_dict.pop('inv_gamma')

        return(slice_data_dict)

    def apply_opmd_standard( self, slice_data_dict, species ):
        """
        Apply the OpenPMD standard to the particle quantities.
        Momentum (u) is multiplied by m * c and weights are
        divided by the particle charge q.
        'Charge' (for ionizable ions) is multiplied by e, and
        thus becomes a float array.

        Parameters
        ----------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")

        species : A ParticleObject
            Contains the particle data and the meta-data
            needed for the conversion to the OpenPMD format.

        Returns
        -------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id")
        """
        # Normalize momenta
        # (only for species that have a mass)
        for quantity in ['ux', 'uy', 'uz']:
            if species.m > 0:
                slice_data_dict[quantity] *= species.m * c
        # Convert ionizable level (integer) to charge in Coulombs (float)
        if 'charge' in slice_data_dict:
            slice_data_dict['charge'] = slice_data_dict['charge']*e

        return slice_data_dict

    def apply_selection( self, select, slice_data_dict ) :
        """
        Apply the rules of self.select to determine which
        particles should be written. Modify the arrays of
        `slice_data_dict` so that only the selected particles remain.

        Parameters
        ----------
        select : a dictionary that defines all selection rules based
        on the quantities

        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")

        Returns
        -------
        slice_data_dict : dictionary of 1D float and integer arrays
            A dictionary that contains the particle data of
            the simulation (with normalized weigths), including optional
            integer arrays (e.g. "id", "charge")
        """
        # Initialize an array filled with True
        N_part_slice = len( slice_data_dict['w'] )
        select_array = np.ones( N_part_slice, dtype='bool' )

        # Apply the rules successively
        # Go through the quantities on which a rule applies
        for quantity in select.keys() :
            # Lower bound
            if select[quantity][0] is not None :
                select_array = np.logical_and( select_array,
                    slice_data_dict[quantity] > select[quantity][0] )
            # Upper bound
            if select[quantity][1] is not None :
                select_array = np.logical_and( select_array,
                    slice_data_dict[quantity] < select[quantity][1] )
        # At this point, `select_array` contains True
        # wherever a particle should be kept

        # Loop through the keys of `select_array` and select only the
        # particles that should be kept.
        for quantity in slice_data_dict.keys():
            slice_data_dict[quantity] = slice_data_dict[quantity][select_array]

        return slice_data_dict


# Alias, for backward compatibility
BoostedParticleDiagnostic = BackTransformedParticleDiagnostic
