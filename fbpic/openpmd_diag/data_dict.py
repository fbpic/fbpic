# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines useful correspondance dictionaries
which are used in the openPMD writer
"""
import numpy as np

# Correspondance between quantity and corresponding dimensions
# As specified in the openPMD standard, the arrays represent the
# 7 basis dimensions L, M, T, I, theta, N, J
unit_dimension_dict = {
    "rho" : np.array([-3., 0., 1., 1., 0., 0., 0.]),
    "J" : np.array([-2., 0., 0., 1., 0., 0., 0.]),
    "E" : np.array([ 1., 1.,-3.,-1., 0., 0., 0.]),
    "Er_pml" : np.array([ 1., 1.,-3.,-1., 0., 0., 0.]),
    "Et_pml" : np.array([ 1., 1.,-3.,-1., 0., 0., 0.]),
    "B" : np.array([ 0., 1.,-2.,-1., 0., 0., 0.]),
    "Br_pml" : np.array([ 0., 1.,-2.,-1., 0., 0., 0.]),
    "Bt_pml" : np.array([ 0., 1.,-2.,-1., 0., 0., 0.]),
    "charge" : np.array([0., 0., 1., 1., 0., 0., 0.]),
    "mass" : np.array([1., 0., 0., 0., 0., 0., 0.]),
    "weighting" : np.array([0., 0., 0., 0., 0., 0., 0.]),
    "position" : np.array([1., 0., 0., 0., 0., 0., 0.]),
    "positionOffset" : np.array([1., 0., 0., 0., 0., 0., 0.]),
    "momentum" : np.array([1., 1.,-1., 0., 0., 0., 0.]),
    "id" : np.array([0., 0., 0., 0., 0., 0., 0.]),
    "gamma" : np.array([0., 0., 0., 0., 0., 0., 0.]) }

# Typical weighting of different particle properties
macro_weighted_dict = {
    "charge": np.uint32(0),
    "mass": np.uint32(0),
    "weighting": np.uint32(1),
    "position": np.uint32(0),
    "positionOffset": np.uint32(0),
    "momentum" : np.uint32(0),
    "E": np.uint32(0),
    "B": np.uint32(0),
    "gamma" : np.uint32(0),
    "id" : np.uint32(0) }
weighting_power_dict = {
    "charge": 1.,
    "mass": 1.,
    "weighting": 1.,
    "position": 0.,
    "positionOffset": 0.,
    "momentum": 1.,
    "E": 0.,
    "B": 0.,
    "gamma": 0.,
    "id": 0. }
