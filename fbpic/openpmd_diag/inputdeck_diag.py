"""

inputdeck_diag.py

opmd derived diagnostic for saving the input deck
into each simulation dump.

"""

import sys
import os
import numpy as np
from fbpic.openpmd_diag.generic_diag import OpenPMDDiagnostic


class InputDeckDiagnostic(OpenPMDDiagnostic):
    """
    Class that allows saving input decks to dumps.
    """
    def __init__(self, period, comm=None, param_dict=None,
                 write_dir=None, iteration_min=0,
                 iteration_max=np.inf, dt_period=None, dt_sim=None ):
        """
        Setup of the input deck diagnostic

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

        # Get the input deck and read it into memory
        self._input_deck = self._get_input_deck()
        self.param_dict = param_dict

    def _get_input_deck(self):
        """
        Try to extract the text from the input deck.
        We assume the input deck was passed as a paramater
        to the currently running script and that it has an
        extension of '.py'.

        Returns
        -------
        The encoded text of the input deck, None if
                no input deck detected
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
        # Return right away if no input deck found
        if self._input_deck is None:
            return

        filename = "data%08d.h5" % iteration
        fullpath = os.path.join(self.write_dir, "hdf5", filename)

        # Open the file again, and set the input deck as attribute
        f = self.open_file(fullpath)
        f.attrs["inputDeck"] = np.string_(self._input_deck)

        # Write the extra parameters, if required
        if self.param_dict is not None:
            for key, val in self.param_dict.items():
                if isinstance(val, str):
                    f.attrs[key] = np.string_(val)
                elif isinstance(val, (list, tuple)):
                    f.attrs[key] = np.array(val)
                else: #if isinstance(val, (int, float)):
                    f.attrs[key] = val

        # Close the file (only the first proc does this)
        if f is not None:
            f.close()
