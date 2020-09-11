# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class BackTransformedFieldDiagnostic

Major features:
- The class reuses the existing methods of FieldDiagnostic
  as much as possible, through class inheritance
- The class implements memory buffering of the slices, so as
  not to write to disk at every timestep
- Parallel output is not implemented for the moment
"""
import os
import numpy as np
from scipy.constants import c
from .field_diag import FieldDiagnostic

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    import cupy
    from fbpic.utils.cuda import cuda, cuda_tpb_bpg_1d, compile_cupy

class BackTransformedFieldDiagnostic(FieldDiagnostic):
    """
    Class that writes the fields *in the lab frame*, from
    a simulation in the boosted frame
    """
    def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshots_lab,
                 Ntot_snapshots_lab, gamma_boost, period, fldobject,
                 comm=None, fieldtypes=["E", "B"],
                 write_dir=None ) :
        """
        Initialize diagnostics that retrieve the data in the lab frame,
        as a series of snapshot (one file per snapshot),
        within a virtual moving window defined by zmin_lab, zmax_lab, v_lab.

        The parameters defined below are specific to the back-transformed
        diagnostics. See the documentation of `FieldDiagnostic` for
        the other parameters.

        Parameters
        ----------
        zmin_lab: float (in meters)
            The position of the left edge of the virtual moving window,
            *in the lab frame*, at t=0
        zmax_lab: float (in meters)
            The position of the right edge of the virtual moving window,
            *in the lab frame*, at t=0

        v_lab: float (in m.s^-1)
            Speed of the moving window *in the lab frame*

        dt_snapshots_lab: float (in seconds)
            Time interval *in the lab frame* between two successive snapshots

        Ntot_snapshots_lab: int
            Total number of snapshots that this diagnostic will produce

        period: int
            Number of iterations for which the data is accumulated in memory,
            before finally writing it to the disk.

        fieldtypes : a list of strings, optional
            The strings are either "rho", "E", "B" or "J"
            and indicate which field should be written.
            Default : only "E" and "B" are written. This is because the
            backward Lorentz transform is not as precise for "rho" and "J" as
            for "E" and "B" (because "rho" and "J" are staggered in time).
            The user can still output "rho" and "J" by changing `fieldtypes`,
            but has to be aware that there may errors in the backward transform.
            Moreover, writing rho/J slows down the simulation, as these fields
            are then brought from spectral to real space, at each iteration.
        """
        # Do not leave write_dir as None, as this may conflict with
        # the default directory ('./diags') in which diagnostics in the
        # boosted frame are written.
        if write_dir is None:
            write_dir='lab_diags'

        # Initialize the normal attributes of a FieldDiagnostic
        FieldDiagnostic.__init__(self, period, fldobject,
                                comm, fieldtypes, write_dir)

        # Register the boost quantities
        self.gamma_boost = gamma_boost
        self.inv_gamma_boost = 1./gamma_boost
        self.beta_boost = np.sqrt( 1. - self.inv_gamma_boost**2 )
        self.inv_beta_boost = 1./self.beta_boost

        # Find the z resolution and size of the diagnostic *in the lab frame*
        # (Needed to initialize metadata in the openPMD file)
        dz_lab = c*self.fld.dt * self.inv_beta_boost*self.inv_gamma_boost
        Nz = int( (zmax_lab - zmin_lab)/dz_lab ) + 1
        self.inv_dz_lab = 1./dz_lab
        # Get number of radial cells in the output
        # (if possible, remove damp cells)
        if self.comm is None:
            Nr = self.fld.interp[0].Nr
        else:
            Nr = self.comm.get_Nr(with_damp=False)

        # Create the list of LabSnapshot objects
        self.snapshots = []
        for i in range( Ntot_snapshots_lab ):
            t_lab = i * dt_snapshots_lab
            snapshot = LabSnapshot( t_lab,
                                    zmin_lab + v_lab*t_lab,
                                    zmax_lab + v_lab*t_lab,
                                    self.write_dir, i, self.fld, Nr )
            self.snapshots.append( snapshot )
            # Initialize a corresponding empty file
            self.create_file_empty_meshes( snapshot.filename, i,
                snapshot.t_lab, Nr, Nz, snapshot.zmin_lab, dz_lab, self.fld.dt)

        # Create a slice handler, which will do all the extraction, Lorentz
        # transformation, etc for each slice to be registered in a LabSnapshot
        self.slice_handler = SliceHandler(self.gamma_boost, self.beta_boost, Nr)

    def write( self, iteration ):
        """
        Redefines the method write of the parent class FieldDiagnostic
        """
        # At each timestep, store a slices of the fields in memory buffers
        self.store_snapshot_slices( iteration )

        # Every self.period, write the buffered slices to disk
        if iteration % self.period == 0:
            self.flush_to_disk()

    def store_snapshot_slices( self, iteration ):
        """
        Store slices of the fields in the memory buffers of the
        corresponding lab snapshots

        Parameter
        ---------
        iteration : int
            The current iteration in the boosted frame simulation
        """
        # If needed: Bring rho/J from spectral space (where they where
        # smoothed/corrected) to real space
        if "rho" in self.fieldtypes or "J" in self.fieldtypes:
            # Get 'rho_prev', since it correspond to rho at time n
            self.fld.spect2interp('rho_prev')
            self.fld.spect2interp('J')
            # Exchange rho and J if needed
            if (self.comm is not None) and (self.comm.size > 1):
                if not self.fld.exchanged_source['J']:
                    self.comm.exchange_fields(self.fld.interp, 'J', 'add')
                if not self.fld.exchanged_source['rho_prev']:
                    self.comm.exchange_fields(self.fld.interp, 'rho', 'add')

        # Find the limits of the local subdomain at this iteration
        if self.comm is None:
            zmin_boost = self.fld.interp[0].zmin
            zmax_boost = self.fld.interp[0].zmax
        else:
            # If a communicator is provided, remove guard and damp cells
            zmin_boost, zmax_boost = self.comm.get_zmin_zmax(
                local=True, with_damp=False, with_guard=False, rank=self.rank )

        # Extract the current time in the boosted frame
        time = iteration * self.fld.dt

        # Loop through the labsnapshots
        for snapshot in self.snapshots:

            # Update the positions of the output slice of this snapshot
            # in the lab and boosted frame (current_z_lab and current_z_boost)
            snapshot.update_current_output_positions( time,
                            self.inv_gamma_boost, self.inv_beta_boost )

            # For this snapshot:
            # - check if the output position *in the boosted frame*
            #   is in the current local domain
            # - check if the output position *in the lab frame*
            #   is within the lab-frame boundaries of the current snapshot
            if ( (snapshot.current_z_boost > zmin_boost) and \
                 (snapshot.current_z_boost < zmax_boost) and \
                 (snapshot.current_z_lab > snapshot.zmin_lab) and \
                 (snapshot.current_z_lab < snapshot.zmax_lab) ):

                # In this case, extract the proper slice from the field array,
                # and store the results into snapshot.slice_array
                # (when running on the GPU, snapshot.slice_array
                # is a device array)
                self.slice_handler.extract_slice(
                    self.fld, self.comm, snapshot.current_z_boost,
                    zmin_boost, snapshot.slice_array )

                # Register snapshot.slice_array in the list of buffers
                # (when running on the GPU, the slice to the CPU)
                snapshot.register_slice( self.inv_dz_lab )

    def flush_to_disk( self ):
        """
        Writes the buffered slices of fields to the disk

        Erase the buffered slices of the LabSnapshot objects
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in self.snapshots:

            # Compact the successive slices that have been buffered
            # over time into a single array, on each proc
            field_array, iz_min, iz_max = snapshot.compact_slices()
            # Perform the Lorentz transformation of the field values
            # *from the boosted frame to the lab frame*, on each proc
            if field_array is not None:
                self.slice_handler.transform_fields_to_lab_frame( field_array )

            # Gather the slices on the first proc
            if self.comm is not None and self.comm.size > 1:
                global_field_array, global_iz_min, global_iz_max = \
                    self.gather_slices( field_array, iz_min, iz_max )
            else:
                global_field_array = field_array
                global_iz_min = iz_min
                global_iz_max = iz_max

            # First proc writes the global array to disk (if it is not empty)
            if (self.rank==0) and (global_field_array is not None):

                # Write to disk
                self.write_slices( global_field_array, global_iz_min,
                    global_iz_max, snapshot, self.slice_handler.field_to_index)

            # Erase the memory buffers
            snapshot.buffered_slices = []
            snapshot.buffer_z_indices = []

    def gather_slices( self, field_array, iz_min, iz_max ):
        """
        Stitch together the field_array of the different processors

        Parameters:
        -----------
        field_array: ndarray of reals, or None
            If the local proc has no slice data, this is None
            Otherwise, it is an array of shape (10, 2*Nm-1, Nr, nslice_local)

        iz_min, iz_max: ints or None
            If the local proc has no slice data, this is None
            Otherwise, it corresponds to the indices at which the data should
            written, in final dataset which is on disk

        Returns:
        --------
        A tuple with:
        global_field_array: an array of shape (10, 2*Nm-1, Nr, nslice_global),
           or None if none of the procs had any data
        global_izmin, global_izmax: the indices at which the global_field_array
           should be written (or None)
        """
        # Gather objects into lists (one element per proc)
        # Note: this is slow, as it uses the generic mpi4py routines gather.
        # (This is because for some proc field_array can be None.)
        mpi_comm = self.comm.mpi_comm
        field_array_list = mpi_comm.gather( field_array )
        iz_min_list = mpi_comm.gather( iz_min )
        iz_max_list = mpi_comm.gather( iz_max )

        # First proc: merge the results
        if self.rank == 0:

            # Check whether any processor had some slices
            no_slices = True
            for f_array in field_array_list:
                if f_array is not None:
                    no_slices = False
                    n_modes = f_array.shape[1] # n_modes is 2*Nm - 1
                    Nr = f_array.shape[2]

            # If there are no slices, set global quantities to None
            if no_slices:
                global_field_array = None
                global_iz_min = None
                global_iz_max = None

            # If there are some slices, gather them
            else:
                # Find the global iz_min and global iz_max
                global_iz_min = min([n for n in iz_min_list if n is not None])
                global_iz_max = max([n for n in iz_max_list if n is not None])

                # Allocate a the global field array, with the proper size
                nslice = global_iz_max - global_iz_min
                data_shape = (10, n_modes, Nr, nslice)
                global_field_array = np.zeros( data_shape )

                # Loop through all the processors
                # Fit the field arrays one by one into the global_field_array
                for i_proc in range(self.comm.size):

                    # If this proc has no data, skip it
                    if field_array_list[ i_proc ] is None:
                        continue
                    # Longitudinal indices within the array global_field_array
                    s_min = iz_min_list[ i_proc ] - global_iz_min
                    s_max = iz_max_list[ i_proc ] - global_iz_min
                    # Copy the array to the proper position
                    global_field_array[:,:,:, s_min:s_max] = \
                                                    field_array_list[i_proc]

            # The first proc returns the result
            return( global_field_array, global_iz_min, global_iz_max )

        # Other processors return a dummy placeholder
        else:
            return( None, None, None )

    def write_slices( self, field_array, iz_min, iz_max, snapshot, f2i ):
        """
        For one given snapshot, write the slices of the
        different fields to an openPMD file

        Parameters
        ----------
        field_array: array of reals
            Array of shape (10, 2*Nm-1, Nr, nslices) which contains
            field values

        iz_min, iz_max: integers
            The indices between which the slices will be written
            iz_min is inclusice and iz_max is exclusive

        snapshot: a LabSnaphot object

        f2i: dict
            Dictionary of correspondance between the field names
            and the integer index in the field_array
        """
        # Open the file without parallel I/O in this implementation
        f = self.open_file( snapshot.filename )

        field_path = "/data/%d/fields/" %snapshot.iteration
        field_grp = f[field_path]

        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes:
            # Scalar field
            if fieldtype == "rho":
                data = field_array[ f2i[ "rho" ] ]
                self.write_field_slices( field_grp, data, "rho",
                                            iz_min, iz_max )
            # Vector field
            elif fieldtype in ["E", "B", "J"]:
                for coord in self.coords:
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "%s/%s" %(fieldtype, coord)
                    data = field_array[ f2i[ quantity ] ]
                    self.write_field_slices( field_grp, data, path,
                                            iz_min, iz_max )

        # Close the file
        f.close()

    def write_field_slices( self, field_grp, data, path, iz_min, iz_max ):
        """
        Writes the slices of a given field into the openPMD file
        """
        dset = field_grp[ path ]
        dset[:, :, iz_min:iz_max ] = data

class LabSnapshot:
    """
    Class that stores data relative to one given snapshot
    in the lab frame (i.e. one given *time* in the lab frame)
    """
    def __init__(self, t_lab, zmin_lab, zmax_lab,
                    write_dir, i, fld, Nr_output):
        """
        Initialize a LabSnapshot

        Parameters
        ----------
        t_lab: float (seconds)
            Time of this snapshot *in the lab frame*

        zmin_lab, zmax_lab: floats
            Longitudinal limits of this snapshot

        write_dir: string
            Absolute path to the directory where the data for
            this snapshot is to be written

        i: int
           Number of the file where this snapshot is to be written

        fld: a Fields object
           This is passed only in order to determine how to initialize
           the slice_array buffer (either on the CPU or GPU)

        Nr_output: int
            Number of cells in the r direction, in the final output
            (This typically excludes the radial damping cells)
        """
        # Deduce the name of the filename where this snapshot writes
        self.filename = os.path.join( write_dir, 'hdf5/data%08d.h5' %i)
        self.iteration = i

        # Time and boundaries in the lab frame (constants quantities)
        self.zmin_lab = zmin_lab
        self.zmax_lab = zmax_lab
        self.t_lab = t_lab

        # Positions where the fields are to be registered
        # (Change at every iteration)
        self.current_z_lab = 0
        self.current_z_boost = 0

        # Buffered field slice and corresponding array index in z
        self.buffered_slices = []
        self.buffer_z_indices = []

        # Allocate a buffer for only one slice (avoids having to
        # reallocate arrays when running on the GPU)
        data_shape = (10, 2*fld.Nm-1, Nr_output)
        if fld.use_cuda is False:
            self.slice_array = np.empty( data_shape )
        else:
            self.slice_array = cupy.empty( data_shape )

    def update_current_output_positions( self, t_boost, inv_gamma, inv_beta ):
        """
        Update the positions of output for this snapshot, so that
        if corresponds to the time t_boost in the boosted frame

        Parameters
        ----------
        t_boost: float (seconds)
            Time of the current iteration, in the boosted frame

        inv_gamma, inv_beta: floats
            Inverse of the Lorentz factor of the boost, and inverse
            of the corresponding beta.
        """
        t_lab = self.t_lab

        # This implements the Lorentz transformation formulas,
        # for a snapshot having a fixed t_lab
        self.current_z_boost = ( t_lab*inv_gamma - t_boost )*c*inv_beta
        self.current_z_lab = ( t_lab - t_boost*inv_gamma )*c*inv_beta

    def register_slice( self, inv_dz_lab ):
        """
        Store the slice of fields represented by self.slice_array
        and also store the z index at which this slice should be
        written in the final lab frame array

        Parameters
        ----------
        inv_dz_lab: float
            Inverse of the grid spacing in z, *in the lab frame*
        """
        # Find the index of the slice in the lab frame
        if self.buffer_z_indices == []:
            # No previous index: caculate it from the absolute z_lab
            iz_lab = int( (self.current_z_lab - self.zmin_lab)*inv_dz_lab )
        else:
            # By construction, this index shoud be the previous index - 1
            # Handling integers avoids unstable roundoff errors, when
            # self.current_z_lab is very close to zmin_lab + iz_lab*dz_lab
            iz_lab = self.buffer_z_indices[-1] - 1

        # Store the array and the index
        self.buffer_z_indices.append( iz_lab )
        # Make a copy of the array if it is directly on the CPU
        if type(self.slice_array) is np.ndarray:
            self.buffered_slices.append( self.slice_array.copy() )
        # or copy from the GPU
        else:
            self.buffered_slices.append( self.slice_array.get() )

    def compact_slices(self):
        """
        Compact the successive slices that have been buffered
        over time into a single array, and return the indices
        at which this array should be written.

        Returns
        -------
        field_array: an array of reals of shape (10, 2*Nm-1, Nr_output, nslices)
        In the above nslices is the number of buffered slices

        iz_min, iz_max: integers
        The indices between which the slices should be written
        (iz_min is inclusive, iz_max is exclusive)

        Returns None if the slices are empty
        """
        # Return None if the slices are empty
        if len(self.buffer_z_indices) == 0:
            return( None, None, None )

        # Check that the indices of the slices are contiguous
        # (This should be a consequence of the transformation implemented
        # in update_current_output_positions, and of the calculation
        # of inv_dz_lab.)
        iz_old = self.buffer_z_indices[0]
        for iz in self.buffer_z_indices[1:]:
            if iz != iz_old - 1:
                raise UserWarning('In the boosted frame diagnostic, '
                        'the buffered slices are not contiguous in z.\n'
                        'The boosted frame diagnostics may be inaccurate.')
                break
            iz_old = iz

        # Pack the different slices together
        # Reverse the order of the slices when stacking the array,
        # since the slices where registered for right to left
        field_array = np.stack( self.buffered_slices[::-1], axis=-1 )

        # Get the first and last index in z
        # (Following Python conventions, iz_min is inclusive,
        # iz_max is exclusive)
        iz_min = self.buffer_z_indices[-1]
        iz_max = self.buffer_z_indices[0] + 1

        return( field_array, iz_min, iz_max )

class SliceHandler:
    """
    Class that extracts, Lorentz-transforms and writes slices of the fields
    """
    def __init__( self, gamma_boost, beta_boost, Nr_output ):
        """
        Initialize the SliceHandler object

        Parameters
        ----------
        gamma_boost, beta_boost: floats
            The Lorentz factor of the boost and the corresponding beta

        Nr_output: int
            Number of cells in the r direction, in the final output
            (This typically excludes the radial damping cells)
        """
        # Store the arguments
        self.gamma_boost = gamma_boost
        self.beta_boost = beta_boost
        self.Nr_output = Nr_output

        # Create a dictionary that contains the correspondance
        # between the field names and array index
        self.field_to_index = {'Er':0, 'Et':1, 'Ez':2, 'Br':3,
                'Bt':4, 'Bz':5, 'Jr':6, 'Jt':7, 'Jz':8, 'rho':9}

    def extract_slice( self, fld, comm, z_boost, zmin_boost, slice_array ):
        """
        Fills `slice_array` with a slice of the fields at z_boost
        (the fields returned are still in the boosted frame ;
        for performance, the Lorentz transform of the fields values
        is performed only when flushing to disk)

        Parameters
        ----------
        fld: a Fields object
            The object from which to extract the fields

        comm: a BoundaryCommunicator object
            Contains information about the gard cells in particular

        z_boost: float (meters)
            Position of the slice in the boosted frame

        zmin_boost: float (meters)
            Position of the left end of physical part of the local subdomain
            (i.e. excludes guard cells)

        slice_array: either a numpy array or a cuda device array
            An array of reals that packs together the slices of the
            different fields (always on array on the CPU).
            The first index of this array corresponds to the field type
            (10 different field types), and the correspondance
            between the field type and integer index is given field_to_index
            The shape of this arrays is (10, 2*Nm-1, Nr_output)
        """
        # Find the index of the slice in the boosted frame
        # and the corresponding interpolation shape factor
        dz = fld.interp[0].dz
        # Find the interpolation data in the z direction
        z_staggered_gridunits = ( z_boost - zmin_boost - 0.5*dz )/dz
        iz = int( z_staggered_gridunits )
        Sz = iz + 1 - z_staggered_gridunits
        # Add the guard cells to the index iz
        if comm is not None:
            iz += comm.n_guard
            if comm.left_proc is None:
                iz += comm.nz_damp+comm.n_inject

        # Extract the slice directly on the CPU
        # Fill the pre-allocated CPU array slice_array
        if fld.use_cuda is False :
            # Extract a slice of the fields *in the boosted frame*
            # at z_boost, using interpolation, and store them in slice_array
            self.extract_slice_cpu( fld, iz, Sz, slice_array )

        # Extract the slice on the GPU
        # Fill the pre-allocated GPU array slice_array
        else:
            # Prepare kernel call
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d( self.Nr_output )

            # Extract the slices
            interp = fld.interp
            for m in range(fld.Nm):
                extract_slice_cuda[ dim_grid_1d, dim_block_1d ](
                    self.Nr_output, iz, Sz, slice_array,
                    interp[m].Er, interp[m].Et, interp[m].Ez,
                    interp[m].Br, interp[m].Bt, interp[m].Bz,
                    interp[m].Jr, interp[m].Jt, interp[m].Jz, interp[m].rho, m)

    def extract_slice_cpu( self, fld, iz, Sz, slice_array ):
        """
        Extract a slice of the fields at iz and iz+1, and interpolated
        between those two points using Sz and (1-Sz)

        Parameters
        ----------
        fld: a Fields object

        iz: int
            Index at which to extract the fields

        Sz: float
            Interpolation shape factor used at iz

        slice_array: np.ndarray
            Array of shape (10, 2*Nm-1, Nr_output )
        """
        # Shortcut for the correspondance between field and integer index
        f2i = self.field_to_index

        # Loop through the fields, and extract the proper slice for each field
        for quantity in self.field_to_index.keys():
            # Here typical values for `quantity` are e.g. 'Er', 'Bz', 'rho'

            # Interpolate the centered field in z
            slice_array[ f2i[quantity], :, : ] = Sz*self.get_dataset(
                                            fld, quantity, iz_slice=iz )
            slice_array[ f2i[quantity], :, : ] += (1.-Sz) * self.get_dataset(
                                            fld, quantity, iz_slice=iz+1 )

    def get_dataset( self, fld, quantity, iz_slice ):
        """
        Extract a given quantity, at a given slice index, from the fields

        Parameters
        ----------
        fld: a Fields object

        quantity: string
            Indicates the quantity to be extracted (e.g. 'Er', 'Bz', 'rho')

        iz_slice: int
            Indicates the position of the slice in the field array

        Returns
        -------
        An array of reals, whose format is close to the final openPMD output.
        In particular, the array of fields is of shape ( 2*Nm-1, Nr)
        (real and imaginary part are separated for each mode)
        """
        # Shortcut
        Nr = self.Nr_output

        # Allocate the array to be returned
        data_shape = (2*fld.Nm-1, Nr)
        output_array = np.empty( data_shape )

        # Get the mode 0 : only the real part is non-zero
        output_array[0,:] = getattr(fld.interp[0], quantity)[iz_slice,:Nr].real
        # Get the higher modes
        # There is a factor 2 here so as to comply with the convention in
        # Lifschitz et al., which is also the convention adopted in FBPIC
        for m in range(1,fld.Nm):
            higher_mode_slice = 2*getattr(fld.interp[m], quantity)[iz_slice,:Nr]
            output_array[2*m-1, :] = higher_mode_slice.real
            output_array[2*m, :] = higher_mode_slice.imag

        return( output_array )

    def transform_fields_to_lab_frame( self, fields ):
        """
        Modifies the array `fields` in place, to transform the field values
        from the boosted frame to the lab frame.

        The transformation is a transformation with -beta_boost, thus
        the corresponding formulas are:
        - for the transverse part of E and B:
        $\vec{E}_{lab} = \gamma(\vec{E} - c\vec{\beta} \times\vec{B})$
        $\vec{B}_{lab} = \gamma(\vec{B} + \vec{\beta}/c \times\vec{E})$
        - for rho and Jz:
        $\rho_{lab} = \gamma(\rho + \beta J_{z}/c)$
        $J_{z,lab} = \gamma(J_z + c\beta \rho)$

        Parameter
        ---------
        fields: array of floats
            An array that packs together the slices of the different fields.
            The shape of this arrays is (10, 2*Nm-1, Nr_output, nslices)
            where nslices is the number of slices that have been buffered
        """
        # Some shortcuts
        gamma = self.gamma_boost
        cbeta = c*self.beta_boost
        beta_c = self.beta_boost/c
        # Shortcut to give the correspondance between field name
        # (e.g. 'Ex', 'rho') and integer index in the array
        f2i = self.field_to_index

        # Lorentz transformations
        # For E and B
        # (NB: Ez and Bz are unchanged by the Lorentz transform)
        # Use temporary arrays when changing Er and Bt in place
        er_lab = gamma*( fields[f2i['Er']] + cbeta * fields[f2i['Bt']] )
        bt_lab = gamma*( fields[f2i['Bt']] + beta_c * fields[f2i['Er']] )
        fields[ f2i['Er'], ... ] = er_lab
        fields[ f2i['Bt'], ... ] = bt_lab
        # Use temporary arrays when changing Et and Br in place
        et_lab = gamma*( fields[f2i['Et']] - cbeta * fields[f2i['Br']] )
        br_lab = gamma*( fields[f2i['Br']] - beta_c * fields[f2i['Et']] )
        fields[ f2i['Et'], ... ] = et_lab
        fields[ f2i['Br'], ... ] = br_lab
        # For rho and J
        # (NB: the transverse components of J are unchanged)
        # Use temporary arrays when changing rho and Jz in place
        rho_lab = gamma*( fields[f2i['rho']] + beta_c * fields[f2i['Jz']] )
        Jz_lab =  gamma*( fields[f2i['Jz']] + cbeta * fields[f2i['rho']] )
        fields[ f2i['rho'], ... ] = rho_lab
        fields[ f2i['Jz'], ... ] = Jz_lab

if cuda_installed:

    @compile_cupy
    def extract_slice_cuda( Nr, iz, Sz, slice_arr,
        Er, Et, Ez, Br, Bt, Bz, Jr, Jt, Jz, rho, m ):
        """
        Extract a slice of the fields at iz and iz+1, and interpolated
        between those two points using Sz and (1-Sz)

        Parameters
        ----------
        Nr: int
            Number of cells transversally

        iz: int
            Index at which to extract the fields

        Sz: float
            Interpolation shape factor used at iz

        slice_arr: cupy.empty
            Array of floats of shape (10, 2*Nm-1, Nr)

        Er, Et, etc...: cupy.empty
            Array of complexs of shape (Nz, Nr), for the azimuthal mode m

        m: int
            Index of the azimuthal mode involved
        """
        # One thread per radial position
        ir = cuda.grid(1)
        # Intermediate variables
        izp = iz+1
        Szp = 1. - Sz

        if ir < Nr:
            # Interpolate the field in the longitudinal direction
            # and store it into pre-packed arrays

            # For the higher modes:
            # There is a factor 2 here so as to comply with the convention in
            # Lifschitz et al., which is also the convention of FBPIC
            # For performance, this is included in the shape factor.
            if m > 0:
                Sz = 2*Sz
                Szp = 2*Szp
                # Index at which the mode should be added
                # in the array `slice_arr`
                im = 2*m-1
            else:
                im = 0

            # Real part
            slice_arr[0,im,ir] = Sz*Er[iz,ir].real + Szp*Er[izp,ir].real
            slice_arr[1,im,ir] = Sz*Et[iz,ir].real + Szp*Et[izp,ir].real
            slice_arr[2,im,ir] = Sz*Ez[iz,ir].real + Szp*Ez[izp,ir].real
            slice_arr[3,im,ir] = Sz*Br[iz,ir].real + Szp*Br[izp,ir].real
            slice_arr[4,im,ir] = Sz*Bt[iz,ir].real + Szp*Bt[izp,ir].real
            slice_arr[5,im,ir] = Sz*Bz[iz,ir].real + Szp*Bz[izp,ir].real
            slice_arr[6,im,ir] = Sz*Jr[iz,ir].real + Szp*Jr[izp,ir].real
            slice_arr[7,im,ir] = Sz*Jt[iz,ir].real + Szp*Jt[izp,ir].real
            slice_arr[8,im,ir] = Sz*Jz[iz,ir].real + Szp*Jz[izp,ir].real
            slice_arr[9,im,ir] = Sz*rho[iz,ir].real + Szp*rho[izp,ir].real

            if m > 0:
                # Imaginary part
                slice_arr[0,im+1,ir] = Sz*Er[iz,ir].imag + Szp*Er[izp,ir].imag
                slice_arr[1,im+1,ir] = Sz*Et[iz,ir].imag + Szp*Et[izp,ir].imag
                slice_arr[2,im+1,ir] = Sz*Ez[iz,ir].imag + Szp*Ez[izp,ir].imag
                slice_arr[3,im+1,ir] = Sz*Br[iz,ir].imag + Szp*Br[izp,ir].imag
                slice_arr[4,im+1,ir] = Sz*Bt[iz,ir].imag + Szp*Bt[izp,ir].imag
                slice_arr[5,im+1,ir] = Sz*Bz[iz,ir].imag + Szp*Bz[izp,ir].imag
                slice_arr[6,im+1,ir] = Sz*Jr[iz,ir].imag + Szp*Jr[izp,ir].imag
                slice_arr[7,im+1,ir] = Sz*Jt[iz,ir].imag + Szp*Jt[izp,ir].imag
                slice_arr[8,im+1,ir] = Sz*Jz[iz,ir].imag + Szp*Jz[izp,ir].imag
                slice_arr[9,im+1,ir] = Sz*rho[iz,ir].imag + Szp*rho[izp,ir].imag


# Alias, for backward compatibility
BoostedFieldDiagnostic = BackTransformedFieldDiagnostic
