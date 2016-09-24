Installation on a local computer
==================================

Installing FBPIC
------------------

The installation requires the
`Anaconda <https://www.continuum.io/why-anaconda>`__ distribution of
Python. If Anaconda is not your default Python distribution, download and install it from `here <https://www.continuum.io/downloads>`__.

-  Clone the ``fbpic`` repository using git.

-  ``cd`` into the top folder of ``fbpic`` and install the dependencies:

   ::

       conda install -c conda-forge --file requirements.txt

-  **Optional:** In order to be able to run the code on a GPU:

   ::

       conda install accelerate

   (The ``accelerate`` package is not free, but there is a 30-day free
   trial period, which starts when the above command is entered. For
   further use beyond 30 days, one option is to obtain an academic
   license, which is also free. To do so, please visit `this
   link <https://www.continuum.io/anaconda-academic-subscriptions-available>`__.)

-  Install ``fbpic``

   ::

       python setup.py install


Testing and potential issues
--------------------------------
       
The installation can be tested by running:

::
   
    pip install openPMD-viewer
    python setup.py test

Please be patient, as the tests can take around 5 minutes to run.

**Potential issues on Mac OSX:** The package ``mpi4py`` can sometimes cause
issues on Mac OSX. If you observe that the code crashes with an
MPI-related error, try installing ``mpi4py`` using MacPorts and
``pip``. To do so, first install `MacPorts <https://www.macports.org/>`_. Then execute the following commands:

::

   conda uninstall mpi4py
   sudo port install openmpi-gcc48
   sudo port select --set mpi openmpi-gcc48-fortran
   pip install mpi4py
    
Running simulations
-------------------

Simulations are run with a user-written python script, which calls the
FBPIC structures. An example script (called ``lpa_sim.py``) can be found
in ``docs/example_input``. The simulation can be run simply by
entering

::

   python lpa_sim.py

The code outputs HDF5 files, that comply with the `OpenPMD
standard <http://www.openpmd.org/#/start>`__, and which can thus be read
as such (e.g. by using the
`openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`__).

For more details on how to set up the input script and run
simulations, see the section :doc:`../how_to_run`.
