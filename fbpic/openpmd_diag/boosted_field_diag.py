"""
This file defines the class BoostedFieldDiagnostic

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

class BoostedFieldDiagnostic(FieldDiagnostic):
    """
    Class that writes the fields *in the lab frame*, from
    a simulation in the boosted frame

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """
    def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshots_lab,
                 Ntot_snapshots_lab, gamma_boost, period, fldobject,
                 comm=None, fieldtypes=["rho", "E", "B", "J"],
                 write_dir=None ) :
        """
        Initialize diagnostics that retrieve the data in the lab frame,
        as a series of snapshot (one file per snapshot),
        within a virtual moving window defined by zmin_lab, zmax_lab, v_lab.

        Parameters
        ----------
        zmin_lab, zmax_lab: floats (meters)
            Positions of the minimum and maximum of the virtual moving window,
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

        See the documentation of FieldDiagnostic for the other parameters
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

        # Create the list of LabSnapshot objects
        self.snapshots = []
        for i in range( Ntot_snapshots_lab ):
            t_lab = i * dt_snapshots_lab
            snapshot = LabSnapshot( t_lab,
                                    zmin_lab + v_lab*t_lab,
                                    zmax_lab + v_lab*t_lab,
                                    self.write_dir, i )
            self.snapshots.append( snapshot )
            # Initialize a corresponding empty file
            self.create_file_empty_meshes( snapshot.filename, i,
                snapshot.t_lab, Nz, snapshot.zmin_lab, dz_lab, self.fld.dt )

        # Create a slice handler, which will do all the extraction, Lorentz
        # transformation, etc for each slice to be registered in a LabSnapshot
        self.slice_handler = SliceHandler( self.gamma_boost, self.beta_boost )

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
        # Find the limits of the local subdomain at this iteration
        zmin_boost = self.fld.interp[0].zmin
        zmax_boost = self.fld.interp[0].zmax
        # If a communicator is provided, remove the guard cells
        if self.comm is not None:
            dz = self.fld.interp[0].dz
            zmin_boost += dz*self.comm.n_guard
            zmax_boost -= dz*self.comm.n_guard

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
                # perform a Lorentz transform to the lab frame, and store
                # the results in a properly-formed array
                slice_array = self.slice_handler.extract_slice(
                    self.fld, self.comm, snapshot.current_z_boost, zmin_boost )
                # Register this in the buffers of this snapshot
                snapshot.register_slice( slice_array, self.inv_dz_lab )

    def flush_to_disk( self ):
        """
        Writes the buffered slices of fields to the disk

        Erase the buffered slices of the LabSnapshot objects
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in self.snapshots:

            # Compact the successive slices that have been buffered
            # over time into a single array
            field_array, iz_min, iz_max = snapshot.compact_slices()
            # Erase the memory buffers
            snapshot.buffered_slices = []
            snapshot.buffer_z_indices = []

            # Write this array to disk (if this snapshot has new slices)
            if field_array is not None:
                self.write_slices( field_array, iz_min, iz_max,
                    snapshot, self.slice_handler.field_to_index )

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
    def __init__(self, t_lab, zmin_lab, zmax_lab, write_dir, i):
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
        """
        # Deduce the name of the filename where this snapshot writes
        self.filename = os.path.join( write_dir, 'hdf5/data%05d.h5' %i)
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

    def register_slice( self, slice_array, inv_dz_lab ):
        """
        Store the slice of fields represented by slice_array
        and also store the z index at which this slice should be
        written in the final lab frame array

        Parameters
        ----------
        slice_array: array of reals
            An array of packed fields that corresponds to one slice,
            as given by the SliceHandler object

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

        # Store the values and the index
        self.buffered_slices.append( slice_array )
        self.buffer_z_indices.append( iz_lab )

    def compact_slices(self):
        """
        Compact the successive slices that have been buffered
        over time into a single array, and return the indices
        at which this array should be written.

        Returns
        -------
        field_array: an array of reals of shape (10, 2*Nm-1, Nr)
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
    def __init__( self, gamma_boost, beta_boost ):
        """
        Initialize the SliceHandler object

        Parameters
        ----------
        gamma_boost, beta_boost: floats
            The Lorentz factor of the boost and the corresponding beta
        """
        # Store the arguments
        self.gamma_boost = gamma_boost
        self.beta_boost = beta_boost

        # Create a dictionary that contains the correspondance
        # between the field names and array index
        self.field_to_index = {'Er':0, 'Et':1, 'Ez':2, 'Br':3,
                'Bt':4, 'Bz':5, 'Jr':6, 'Jt':7, 'Jz':8, 'rho':9}

    def extract_slice( self, fld, comm, z_boost, zmin_boost ):
        """
        Returns an array that contains the slice of the fields at
        z_boost (the fields returned are already transformed to the lab frame)

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

        Returns
        -------
        An array of reals that packs together the slices of the
        different fields.

        The first index of this array corresponds to the field type
        (10 different field types), and the correspondance
        between the field type and integer index is given self.field_to_index

        The shape of this arrays is (10, 2*Nm-1, Nr)
        """
        # Extract a slice of the fields *in the boosted frame*
        # at z_boost, using interpolation, and store them in an array
        # (See the docstring of the extract_slice_boosted_frame for
        # the shape of this array.)
        slice_array = self.extract_slice_boosted_frame(
                            fld, comm, z_boost, zmin_boost )

        # Perform the Lorentz transformation of the fields *from
        # the boosted frame to the lab frame*
        self.transform_fields_to_lab_frame( slice_array )

        return( slice_array )

    def extract_slice_boosted_frame( self, fld, comm, z_boost, zmin_boost ):
        """
        Extract a slice of the fields at z_boost, using interpolation in z

        See the docstring of extract_slice for the parameters.

        Returns
        -------
        An array that packs together the slices of the different fields.
            The shape of this arrays is (10, 2*Nm-1, Nr)
        """
        # Allocate an array of the proper shape
        slice_array = np.empty( (10, 2*fld.Nm-1, fld.Nr) )

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

        # Shortcut for the correspondance between field and integer index
        f2i = self.field_to_index

        # Loop through the fields, and extract the proper slice for each field
        for quantity in self.field_to_index.keys():
            # Here typical values for `quantity` are e.g. 'Er', 'Bz', 'rho'

            # Interpolate the centered field in z
            # (Transversally-staggered fields are also interpolated
            # to the nodes of the grid, thanks to the flag transverse_centered)
            slice_array[ f2i[quantity], :, : ] = Sz*self.get_dataset(
                                            fld, quantity, iz_slice=iz )
            slice_array[ f2i[quantity], ... ] += (1.-Sz) * self.get_dataset(
                                            fld, quantity, iz_slice=iz+1 )

        return( slice_array )

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
        # Allocate the array to be returned
        data_shape = ( 2*fld.Nm-1, fld.Nr )
        output_array = np.empty( data_shape )

        # Get the mode 0 : only the real part is non-zero
        output_array[0,:] = getattr(fld.interp[0], quantity)[iz_slice,:].real
        # Get the higher modes
        # There is a factor 2 here so as to comply with the convention in
        # Lifschitz et al., which is also the convention adopted in Warp Circ
        for m in range(1,fld.Nm):
            higher_mode_slice = 2*getattr(fld.interp[m], quantity)[iz_slice,:]
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
            The shape of this arrays is (10, 2*Nm-1, Nr)
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
        rho_lab = gamma*( fields[f2i['rho']] + 0*beta_c * fields[f2i['Jz']] )
        Jz_lab =  gamma*( fields[f2i['Jz']] + 0*cbeta * fields[f2i['rho']] )
        fields[ f2i['rho'], ... ] = rho_lab
        fields[ f2i['Jz'], ... ] = Jz_lab
