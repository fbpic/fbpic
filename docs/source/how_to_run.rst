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

The different FBPIC objects that are used in the scripts are defined
in the section :doc:`api_reference/api_reference`. To understand how
the script work, you can also use the tutorial below, which goes through
one script step by step.
  
The simulation is then run by typing

::

   python fbpic_script.py
   
where ``fbpic_script.py`` should be replaced by the name of your
Python script: either ``lwfa_script.py`` or
``boosted_frame_script.py`` for the above examples.



Tutorial
-----------

