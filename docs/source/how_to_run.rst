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

The different FBPIC objects that are used in the simulation scripts are defined
in the section :doc:`api_reference/api_reference`.

Running and visualizing the simulation
-----------------------------------------

The simulation is then run by typing

::

   python fbpic_script.py
   
where ``fbpic_script.py`` should be replaced by the name of your
Python script: either ``lwfa_script.py`` or
``boosted_frame_script.py`` for the above examples.

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
