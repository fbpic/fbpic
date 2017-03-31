# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It contains a helper function that parses the data file atomic_data.txt
"""
import re, os
import numpy as np
from scipy.constants import e

cashed_ionization_energies = {}

def get_ionization_energies( element ):
    """
    Return an array of ionization energies (in Joules), with one
    array element per ionization state.

    If the same element was requested previously, the ionization energy
    is obtained from a cashed dictionary (`cashed_ionization_energies`)
    otherwise the energies are read from a data file.

    Parameters
    ----------
    element: string
        The atomic symbol of the considered ionizable species
        (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

    Returns
    -------
    An array with one array element per ionization state, containing the
    ionization energy in Joules, or None if the element was not found in
    the file.
    """
    # Lookup in the cashed dictionary
    if element in cashed_ionization_energies.keys():
        print('Getting cashed ionization energy for %s.' %element )
        return( cashed_ionization_energies[element] )
    else:
        energies = read_ionization_energies(element)
        # Record energies in the cashed dictionary
        cashed_ionization_energies[element] = energies
        return( energies )

def read_ionization_energies( element ):
    """
    Read the ionization energies from a data file

    Parameters
    ----------
    element: string
        The atomic symbol of the considered ionizable species
        (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

    Returns
    -------
    An array with one array element per ionization state, containing the
    ionization energy in Joules.
    """
    # Open and read the file atomic_data.txt
    filename = os.path.join( os.path.dirname(__file__), 'atomic_data.txt' )
    with open(filename) as f:
        text_data = f.read()
    # Parse the data using regular expressions (a.k.a. regex)
    # (see https://docs.python.org/2/library/re.html)
    # The regex command below parses lines of the type
    # '\n     10 | Ne IV         |         +3 |           [97.1900]'
    # and only considers those for which the element (Ne in the above example)
    # matches the element which is passed as argument of this function
    # For each line that satisfies this requirement, it extracts a tuple with
    # - the atomic number (represented as (\d+))
    # - the ionization level (represented as the second (\d+))
    # - the ionization energy (represented as (\d+\.*\d*))
    regex_command = \
        '\n\s+(\d+)\s+\|\s+%s\s+\w+\s+\|\s+\+*(\d+)\s+\|\s+\(*\[*(\d+\.*\d*)' \
        %element
    list_of_tuples = re.findall( regex_command, text_data )
    # Return None if the requested element was not found
    if list_of_tuples == []:
        return(None)
    # Go through the list of tuples and fill the array of ionization energies.
    atomic_number = int( list_of_tuples[0][0] )
    assert atomic_number > 0
    energies = np.zeros( atomic_number )
    for ion_level in range( atomic_number ):
        # Check that, when reading the file,
        # we obtained the correct ionization level
        assert ion_level == int( list_of_tuples[ion_level][1] )
        # Get the ionization energy and convert in Joules using e
        energies[ ion_level ] = e * float( list_of_tuples[ion_level][2] )

    return( energies )
