# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Alberto de la Ossa
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the mirror class, which set the fields to 0 in a
thin slice orthogonal to z
"""
from scipy.constants import c


class Mirror(object):

    def __init__( self, z_start, z_end, gamma_boost=None, m='all'):
        """
        Initialize a mirror.

        The mirror reflects the fields in the z direction, by setting the
        specified field modes to 0 in a thin slice orthogonal to z, at each timestep.s
        By default, all modes are zeroed.

        Parameters
        ----------
        z_start: float
            Start position of the mirror in the lab frame

        z_end: float
            End position of the mirror in the lab frame

        gamma_boost: float
            For boosted-frame simulation: Lorentz factor of the boost

        m: int or list of ints
            Specify the field modes to set to zero
            By default, takes all modes to zero
        """
        
        self.z_start = z_start
        self.z_end = z_end
        self.gamma_boost = gamma_boost

        if m == 'all':
            self.modes = None
        elif isinstance(m, int):
            self.modes = [m]
        elif isinstance(m, list):
            self.modes = m
        else:
            raise TypeError('m should be an int or a list of ints.')

    def set_fields_to_zero( self, interp, comm, t_boost):
        """
        Set the fields to 0 in a slice orthogonal to z

        Parameters:
        -----------
        interp: a list of InterpolationGrid objects
            Contains the values of the fields in interpolation space
        comm: a BoundaryCommunicator object
            Contains information on the position of the mesh
        t_boost: float
            Time in the boosted frame
        """
        # Lorentz transform
        if self.gamma_boost is None:
            z_start_boost, z_end_boost = self.z_start, self.z_end
        else:
            beta_boost = (1. - 1. / self.gamma_boost**2)**.5
            z_start_boost = 1. / self.gamma_boost * self.z_start - beta_boost * c * t_boost
            z_end_boost = 1. / self.gamma_boost * self.z_end - beta_boost * c * t_boost

        # Calculate indices in z between which the field should be set to 0
        zmin, zmax = comm.get_zmin_zmax( local=True,
                        with_guard=True, with_damp=True, rank=comm.rank)
        if (z_start_boost < zmin) or (z_start_boost >= zmax):
            return

        imax = int( (z_start_boost - zmin) / interp[0].dz)
        n_cells = int( (z_end_boost - z_start_boost) / interp[0].dz)
        imin = max( imax - n_cells, 0)

        # Set fields (E, B) to 0 on CPU or GPU
        for i, grid in enumerate(interp):

            if self.modes is not None:
                if i not in self.modes:
                    continue
            
            fieldlist = ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz']
            if grid.use_pml:
                fieldlist = fieldlist + ['Er_pml', 'Et_pml', 'Br_pml', 'Bt_pml']
            for field in fieldlist:
                arr = getattr( grid, field)
                arr[ imin:imax, :] = 0.  # Uses numpy/cupy syntax
