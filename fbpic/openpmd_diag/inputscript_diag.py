# Copyright 2019, FBPIC contributors
# Authors: Kris Poder
# License: 3-Clause-BSD-LBNL
"""
This file defines the class InputScriptDiagnostic.

It allows saving the input script into each simulation dump.
"""

import sys
import os
import numpy as np
from fbpic.openpmd_diag.generic_diag import OpenPMDDiagnostic


class InputScriptDiagnostic(OpenPMDDiagnostic):
    """
    Class that allows saving input decks to dumps.
    """
    def __init__(self, period, comm=None, param_dict=None,
                 write_dir=None, iteration_min=0,
                 iteration_max=np.inf, dt_period=None, dt_sim=None ):
        """
        Setup of the input script diagnostic

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc
            Otherwise, each proc writes its own data.
            (Make sure to use different write_dir in this case.)

        param_dict : a dictionary of additional attributes
            If this is not None, the dictionary entries will be written
            to each file dump. The keys of the dict should be in camelCase.

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        dt_sim : float (in seconds), optional
            The timestep of the simulation.
            Only needed if `dt_period` is not None.
        """
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                                   iteration_min, iteration_max,
                                   dt_period=dt_period, dt_sim=dt_sim)

        # Get the input `script` and read it into memory
        self._input_script = self._get_input_script()
        self.param_dict = param_dict

    def _get_input_script(self):
        """
        Try to extract the text from the input script.
        We assume the input script was passed as a paramater
        to the currently running script and that it has an
        extension of '.py'.

        Returns
        -------
        The encoded text of the input script, None if
                no input script detected
        """
        args = sys.argv
        py_files = [f for f in args if '.py' in f]
        if not py_files:
            return
        with open(py_files[0], 'r') as f:
            text = f.read()
        return text

    def write_hdf5(self, iteration):
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Return right away if no input script found
        if self._input_script is None:
            return

        filename = "data%08d.h5" % iteration
        fullpath = os.path.join(self.write_dir, "hdf5", filename)

        # Open the file again, and set the input script as attribute
        f = self.open_file(fullpath)
        # Only the first proc does writing to file
        if f is not None:
            f.attrs["inputScript"] = np.string_(self._input_script)

            # Write the extra parameters, if required
            if self.param_dict is not None:
                for key, val in self.param_dict.items():
                    if isinstance(val, str):
                        f.attrs[key] = np.string_(val)
                    elif isinstance(val, (list, tuple)):
                        f.attrs[key] = np.array(val)
                    else: #if isinstance(val, (int, float)):
                        f.attrs[key] = val

            f.close()
