How to run the code
===================

Once installed (see :doc:`install/installation`), FBPIC is available as a **Python
module** on your system. Thus, a simulation is setup by creating a
**Python script** that imports and uses FBPIC's functionalities.

Script examples
----------------

You can download examples of FBPIC scripts below (which you can then modify
to suit your needs):

- Standard simulation of laser-wakefield acceleration: :download:`lwfa_script.py <example_input/lwfa_script.py>`
- Boosted-frame simulation of laser-wakefield acceleration: :download:`boosted_frame_script.py <example_input/boosted_frame_script.py>`

(See the section :doc:`advanced/boosted_frame` for more information on the above script.)

The different FBPIC objects that are used in the simulation scripts are defined
in the section :doc:`api_reference/api_reference`.

Running the simulation
----------------------

The simulation is then run by typing

::

   python fbpic_script.py

where ``fbpic_script.py`` should be replaced by the name of your
Python script: either ``lwfa_script.py`` or
``boosted_frame_script.py`` for the above examples.

.. note::

   When running on CPU, **multi-threading** is enabled by default, and the
   default number of threads is the number of cores on your system. You
   can modify this with environment variables:

   - To modify the number of threads (e.g. set it to 8 threads):

   ::

	export MKL_NUM_THREADS=8
	export NUMBA_NUM_THREADS=8
	python fbpic_script.py

   - To disable multi-threading altogether (except for the FFT and DHT):

   ::

	export FBPIC_DISABLE_THREADING=1
	python fbpic_script.py

Visualizing the simulation results
----------------------------------

The code outputs HDF5 files, that comply with the
`openPMD standard <http://www.openpmd.org/#/start>`_. As such, they
can be visualized for instance with the `openPMD-viewer
<https://github.com/openPMD/openPMD-viewer>`_). To do so, first
install the openPMD-viewer by typing

::

   conda install -c rlehe openpmd_viewer

And then type

::

   openPMD_notebook

and follow the instructions in the notebook that pops up. (NB: the
notebook only shows some of the capabilities of the openPMD-viewer. To
learn more, see the tutorial notebook on the  `Github repository
<https://github.com/openPMD/openPMD-viewer>`_ of openPMD-viewer).
