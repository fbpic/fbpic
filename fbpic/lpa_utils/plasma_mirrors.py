# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the plasma mirror class, which set the fields to 0 in a
thin slice orthogonal to z
"""
from scipy.constants import c

class PlasmaMirror(object):

    def __init__( self, z_lab, n_cells=2, gamma_boost=None ):
        """
        TODO

        z_lab: position of the plasma mirror in the lab frame
        """
        self.z_lab = z_lab
        self.gamma_boost = gamma_boost
        self.n_cells = n_cells

        pass

    def set_fields_to_zero( self, interp, comm, t_boost ):
        """
        Set the fields to 0 in a slice orthogonal to z

        t_boosts: time in the boosted-frame
        """
        # Lorentz transform
        if self.gamma_boost is None:
            z_boost = self.z_lab
        else:
            beta_boost = (1. - 1./self.gamma_boost**2)**.5
            z_boost = 1./self.gamma_boost*self.z_lab - beta_boost * c * t_boost

        # Calculate indices in z between which the field should be set to 0
        zmin, zmax = comm.get_zmin_zmax( local=True,
                        with_guard=True, with_damp=True, rank=comm.rank )
        if (z_boost < zmin) or (z_boost >= zmax):
            return
        imax = int( (z_boost-zmin)/interp[0].dz )
        imin = max( imax-self.n_cells, 0 )

        # Set fields (E, B) to 0 on CPU or GPU
        for grid in interp:
            for field in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz']:
                arr = getattr( grid, field )
                arr[ imin:imax, : ] = 0.  # Uses numpy/cupy syntax
