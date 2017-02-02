# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with particle tracking.
"""
import numpy as np
from numba import cuda

class ParticleTracker(object):
    """
    # TODO
    """
    def __init__( self, comm_size, comm_rank, N, use_cuda ):
        """
        # TODO
        """
        # Prepare how to attribute new ids
        self.next_attributed_id = comm_rank
        self.id_step = comm_size
        # Everytime a new id is attributed, next_attributed_id is incremented
        # by id_step ; this way, all the particles (even across different
        # MPI proc) have unique id.

        # Initialize the array of ids
        new_next_attributed_id = self.next_attributed_id + N*self.id_step
        self.id = np.arange(
            start=self.next_attributed_id, stop=new_next_attributed_id,
            step=self.id_step, dtype=np.uint64 )
        self.next_attributed_id = new_next_attributed_id

        # Create a sorting buffer
        self.sorting_buffer = np.empty( N, dtype=np.uint64 )

    def send_to_gpu(self):
        """
        # TODO
        """
        self.id = cuda.to_device( self.id )
        self.sorting_buffer = cuda.to_device( self.sorting_buffer )

    def receive_from_gpu(self):
        """
        # TODO
        """
        self.id = self.id.copy_to_host()
        self.sorting_buffer = self.sorting_buffer.copy_to_host()

    def generate_new_ids( self, N ):
        """
        # TODO
        """
        new_next_attributed_id = self.next_attributed_id + N*self.id_step
        new_ids = np.arange(
            start=self.next_attributed_id, stop=new_next_attributed_id,
            step=self.id_step, dtype=np.uint64 )
        self.next_attributed_id = new_next_attributed_id
        return( new_ids )

    def rewrite_ids( self, pid, comm ):
        """
        # TODO
        """
        # Get the new ids
        self.id[:] = pid

        # Set self.next_attributed_id, so that attributed ids are still unique
        # In order to do this, find the maximum of all pid across processors
        if len(pid) > 0:
            local_id_max = pid.max()
        else:
            local_id_max = 0
        if comm.mpi_comm is None:
            global_id_max = local_id_max
        else:
            local_id_max_list = comm.mpi_comm.allgather( local_id_max )
            global_id_max = max( local_id_max_list )
        # Find the next_attibuted_id: has to be of the form
        # comm.rank + n*self.id_step
        n = int( (global_id_max - comm.rank)/self.id_step ) + 1
        self.next_attibuted_id = comm.rank + n*self.id_step
