# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL

"""
This file defines the class PhaseSpaceDiagnostic
"""
import os
import h5py
import numpy as np
from scipy import constants
from .generic_diag import OpenPMDDiagnostic
from .data_dict import macro_weighted_dict, weighting_power_dict

class PhaseSpaceDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines a phase space diagnostic to be performed.
    """

    def __init__(self, period=None, species={}, comm=None,
        name=None, phase_space=[], bins=[], edges=None, 
        move_with_window=True, deposit='w', select=None, write_dir=None, 
        iteration_min=0, iteration_max=np.inf, dt_period=None ) :
        """
        Initialize a phase space diagnostic.

        Parameters
        ----------
        period : int, optional
            The period of the diagnostic, in number of timesteps.
            (i.e. the diagnostic is written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        dt_period : float (in seconds), optional
            The period of the diagnostic, in physical time of the simulation.
            Specify either this or `period`

        species : a dictionary of :any:`Particles` objects
            The object that is written (e.g. elec)
            is assigned to the particle name of this species.
            (e.g. {"electrons": elec })

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered by the communicator, and
            guard cells are removed.
            Otherwise, each rank writes its own data, including guard cells.
            (Make sure to use different write_dir in this case.)
        
        name: str, optional
            Specify a custom name for the diagnostic. If left as None, an
            automatically generated name is used based on the phase space
            and quantity deposited
            
        phase_space : a list of strings
            Specify the phase space over which to bin particles, of the form
            ['uz']        (bin along the z momentum)
            ['z','gamma'] (bin along z position and the particle gamma)
            
        bins: a list of ints
            A list of ints specifying the number of bins in each dimension.

        edges: list, optional
            Set the outer bin edges for each dimension.
            Leave as None to auto-scale, or set pairs of floats to specify 
            limits, of the form.
            [[-1,1], None]  (set a range of (-1, 1) in the first dimension, 
                             and auto-scale the second)
            When using a moving window, any limits for binning along z 
            will automatically be shifted along with the box, this behaviour
            can be turned off with the move_with_window argument.
        
        move_with_window: bool, optional
            Move any given z limits along with the moving window
            
        deposit: string, optional
            Specify a particle quantity to deposit in the bins. 
            The particle weight is always deposited.
                    
        select : dict, optional
            Either None or a dictionary of rules
            to select the particles, of the form
            'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
            'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
            'uz' : [5., None]  (Particles with uz above 5 mc)

        write_dir : a list of strings, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        """
        # Check input
        if len(species) == 0:
            raise ValueError("You need to pass an non-empty `species_dict`.")
        if len(phase_space) == 0:
            raise ValueError("You need to pass a non-empty `phase_space`")
        # Check the dimensions of the arguments
        if len(bins) != len(phase_space):
            raise ValueError("Dimensionality of the phase space and the bins must be equal.")
        if edges is not None:
            if len(edges) != len(phase_space):
                raise ValueError("Any edge limits specified must match the phase space dimensionality")  

                
                
        # Build an ordered list of species. (This is needed since the order
        # of the keys is not well defined, so each MPI rank could go through
        # the species in a different order, if species_dict.keys() is used.)
        self.species_names_list = sorted( species.keys() )
        # Extract the timestep from the first species
        first_species = species[self.species_names_list[0]]
        self.dt = first_species.dt
                                              
        # General setup (uses the above timestep)
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                        iteration_min, iteration_max,
                        dt_period=dt_period, dt_sim=self.dt )

        # Register the arguments
        self.species_dict = species
        self.name = name
        self.phase_space = phase_space
        self.select = select
        self.bins = bins
        self.edges = edges
        self.deposit = deposit
        self.move_with_window = move_with_window
        
        # Register the dimensionality of the diagnostic
        self.Ndims = len(phase_space)
        
        # For each species, get the particle arrays to be written
        self.constant_quantities_dict = {}
        self.array_quantities_dict = {}

        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            # Get the list of the required particle quantities
            self.array_quantities_dict[species_name] = phase_space

            #self.array_quantities_dict[species_name].append(phase_space)

            # Get the list of quantities that are constant
            self.constant_quantities_dict[species_name] = ["mass"]
            # For ionizable particles, the charge must be treated as an array
            if species.ionizer is not None:
                self.array_quantities_dict[species_name] += ["charge"]
            else:
                self.constant_quantities_dict[species_name] += ["charge"]
                

    def setup_openpmd_species_group( self, grp, species, constant_quantities ):
        """
        Set the attributes that are specific to the particle group

        Parameter
        ---------
        grp : an h5py.Group object
            Contains all the species

        species : a fbpic Particle object

        constant_quantities: list of strings
            The scalar quantities to be written for this particle
        """
        # Generic attributes
        grp.attrs["particleShape"] = 1.
        grp.attrs["currentDeposition"] = np.string_("directMorseNielson")
        grp.attrs["particleSmoothing"] = np.string_("none")
        grp.attrs["particlePush"] = np.string_("Vay")
        grp.attrs["particleInterpolation"] = np.string_("uniform")

        # Setup constant datasets (e.g. charge, mass)
        for quantity in constant_quantities:
            grp.require_group( quantity )
            self.setup_openpmd_species_record( grp[quantity], quantity )
            self.setup_openpmd_species_component( grp[quantity], quantity )
            grp[quantity].attrs["shape"] = np.array([1], dtype=np.uint64)
        # Set the corresponding values
        grp["mass"].attrs["value"] = species.m
        if "charge" in constant_quantities:
            grp["charge"].attrs["value"] = species.q

        # Set the position records (required in openPMD)
        quantity = "positionOffset"
        grp.require_group(quantity)
        self.setup_openpmd_species_record( grp[quantity], quantity )
        for quantity in [ "positionOffset/x", "positionOffset/y",
                            "positionOffset/z"] :
            grp.require_group(quantity)
            self.setup_openpmd_species_component( grp[quantity], quantity )
            grp[quantity].attrs["shape"] = np.array([1], dtype=np.uint64)

        # Set the corresponding values
        grp["positionOffset/x"].attrs["value"] = 0.
        grp["positionOffset/y"].attrs["value"] = 0.
        grp["positionOffset/z"].attrs["value"] = 0.

    def setup_openpmd_species_record( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species record

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
            The group that correspond to `quantity`
            (in particular, its path must end with "/<quantity>")

        quantity : string
            The name of the record being setup
            e.g. "position", "momentum"
        """
        # Generic setup

        self.setup_openpmd_record( grp, quantity )

        # Weighting information
        grp.attrs["macroWeighted"] = macro_weighted_dict[quantity]
        grp.attrs["weightingPower"] = weighting_power_dict[quantity]

    def setup_openpmd_species_component( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species component

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset

        quantity : string
            The name of the component
        """
        self.setup_openpmd_component( grp )

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Receive data from the GPU if needed
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            if species.use_cuda :
                species.receive_particles_from_gpu()

        # Create the file and setup the openPMD structure (only first proc)
        if self.rank == 0:
            filename = "data%08d.h5" %iteration
            fullpath = os.path.join( self.write_dir, "hdf5", filename )
            f = h5py.File( fullpath, mode="a" )

            # Setup its attributes
            self.setup_openpmd_file( f, iteration, iteration*self.dt, self.dt)

        # Loop over the different species and
        # particle quantities that should be written
        for species_name in self.species_names_list:

            # Check if the species exists
            species = self.species_dict[species_name]
            if species is None :
                # If not, immediately go to the next species_name
                continue

            # Setup the species group (only first proc)
            if self.rank==0:
                species_path = "/data/%d/phase spaces/%s" %(
                    iteration, species_name)
                # Create and setup the h5py.Group species_grp
                species_grp = f.require_group( species_path )
                self.setup_openpmd_species_group( species_grp, species,
                                self.constant_quantities_dict[species_name])
            else:
                species_grp = None

            # Select the particles that will be written
            select_array = self.apply_selection( species )
            # Get their total number
            n = select_array.sum()
            if self.comm is not None:
                # Multi-proc output
                if self.comm.size > 1:
                    n_rank = self.comm.mpi_comm.allgather(n)
                else:
                    n_rank = [n]
                Ntot = sum(n_rank)
            else:
                # Single-proc output
                n_rank = None
                Ntot = n
            
            # Write the datasets for each particle datatype
            self.write_phasespaces( species_grp, species, n_rank, Ntot, 
                                   select_array, 
                                   self.array_quantities_dict[species_name] )

        # Close the file
        if self.rank == 0:
            f.close()

        # Send data to the GPU if needed
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            if species.use_cuda :
                species.send_particles_to_gpu()

    def write_phasespaces( self, species_grp, species, n_rank,
                           Ntot, select_array, phasespace_data ) :
        """
        Write all the phase space sets for one given species

        species_grp : an h5py.Group
            The group where to write the species considered

        species : an fbpic.Particles object
        	The species object to get the particle data from

        n_rank : list of ints
            A list containing the number of particles to send on each proc

        Ntot : int
        	Contains the global number of particles

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select

        phasespace_data: list of string
            The phasespace quantities that should be written
        """
        # Set the dataset name if one was provided
        if self.name is not None:
            quantity_path = self.name
        else:
            # Otherwise automatically generate a name
            quantity_path = '%s_%s'%('_'.join(self.phase_space), self.deposit)
        
        self.write_dataset( species_grp, species, quantity_path,
                             n_rank, Ntot, select_array )


    def apply_selection( self, species ) :
        """
        Apply the rules of self.select to determine which
        particles should be written, Apply random subsampling using
        the property subsampling_fraction.

        Parameters
        ----------
        species : a Species object

        Returns
        -------
        A 1d array of the same shape as that particle array
        containing True for the particles that satify all
        the rules of self.select
        """
        # Initialize an array filled with True
        select_array = np.ones( species.Ntot, dtype='bool' )


        # Apply the rules successively
        if self.select is not None :
            # Go through the quantities on which a rule applies
            for quantity in self.select.keys() :
                if quantity == "gamma":
                    quantity_array = 1.0/getattr( species, "inv_gamma" )
                else:
                    quantity_array = getattr( species, quantity )
                # Lower bound
                if self.select[quantity][0] is not None :
                    select_array = np.logical_and(
                        quantity_array > self.select[quantity][0],
                        select_array )
                # Upper bound
                if self.select[quantity][1] is not None :
                    select_array = np.logical_and(
                        quantity_array < self.select[quantity][1],
                        select_array )

        return( select_array )


    def write_dataset( self, species_grp, species, path,
                       n_rank, Ntot, select_array ) :
        """
        Write a given dataset

        Parameters
        ----------
        species_grp : an h5py.Group
            The group where to write the phasespace considered

        species : a fbpic.Particles object
        	The species object to get the particle data from

        path : string
            The relative path where to write the dataset,
            inside the species_grp

        n_rank : list of ints
            A list containing the number of particles to send on each proc

        Ntot : int
        	Contains the global number of particles

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
 
        # Create the datasets and set up their attributes
        if self.rank==0:
            datashape = self.bins
            dtype = 'f8'
            # If the dataset already exists, remove it.
            # (This avoids errors with diags from previous simulations,
            # in case the number of particles is not exactly the same.)
            if path in species_grp:
                del species_grp[path]
            
            dset = species_grp.create_dataset(path, datashape, dtype=dtype )

        # Set up the sample array
        sample = np.empty((Ntot, self.Ndims))
    
        # contruct the sample to be binned
        for i in range(self.Ndims):
            sample[:,i] = self.get_dataset( species, self.phase_space[i], 
                                           select_array, n_rank, Ntot ) 
            
        # Set up a temporary list for the bin edges
        bin_edges = []     
        # Build the bins edges from the provided limits and deal with the
        # moving window if required
        if self.edges is None:
            # If none are specified, allow histogramdd to auto-scale
            bin_edges = None
        else:
            # Else, work through the provided bin edges
            for i in range(self.Ndims):          
                # If the phase space being binned is z, and there is a moving
                # window set up, and the user has not specified otherwise,
                # adjust the limits
                if all([self.phase_space[i] == 'z', self.move_with_window, 
                       self.comm.moving_win is not None]):
                    t = self.comm.moving_win.t_last_move
                    v = self.comm.moving_win.v  
                    bin_edges.append([z+v*t for z in self.edges[i]])
                else:
                    # Otherwise use the edges as they are
                    bin_edges.append(self.edges[i])

        # Get the particle weights
        weights = self.get_dataset( species, 'w', select_array,
                                           n_rank, Ntot )
        
        # deposit a specified quantity alongside the weights
        if self.deposit != 'w':
            deposit = self.get_dataset( species, self.deposit, select_array, 
                                        n_rank, Ntot )            
            weights *= deposit
            # set the deposition metadata
            dset.attrs['deposit'] = self.deposit 
        else:
            dset.attrs['deposit'] = 'weight'        
  
        # generate the histogram
        hist, edges = np.histogramdd(sample, bins=self.bins, 
                                        density=False, weights=weights,
                                        range=bin_edges)
        
        # Fill out the dataset and metadata for the axes
        if self.rank==0:
            dset[:] = hist
            for i in range(self.Ndims):
                dset.attrs['axis%i'%(i+1)] = [edges[i][0], edges[i][-1]]
                dset.attrs['axis%iName'%(i+1)] = '%s'%self.phase_space[i]


    def get_dataset( self, species, quantity, select_array, n_rank, Ntot ) :
        """
        Extract the array that satisfies select_array

        species : a Particles object
        	The species object to get the particle data from

        quantity : string
            The quantity to be extracted (e.g. 'x', 'uz', 'w')

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select

        n_rank: list of ints
        	A list containing the number of particles to send on each proc

        Ntot : int
            Length of the final array (selected + gathered from all proc)
        """
        # Extract the quantity
        if quantity == "id":
            quantity_one_proc = species.tracker.id
        elif quantity == "charge":
            quantity_one_proc = constants.e * species.ionizer.ionization_level
        elif quantity == "w":
            quantity_one_proc = species.w
        elif quantity == "gamma":
            quantity_one_proc = 1.0/getattr( species, "inv_gamma" )
        else:
            quantity_one_proc = getattr( species, quantity )

        # Apply the selection
        quantity_one_proc = quantity_one_proc[ select_array ]

        # If this is the momentum, multiply by the proper factor
        # (only for species that have a mass)
        if quantity in ['ux', 'uy', 'uz']:
            if species.m>0:
                scale_factor = species.m * constants.c
                quantity_one_proc *= scale_factor
        if self.comm is not None:
            quantity_all_proc = self.comm.gather_ptcl_array(
                quantity_one_proc, n_rank, Ntot )
        else:
            quantity_all_proc = quantity_one_proc

        # Return the results
        return( quantity_all_proc )
