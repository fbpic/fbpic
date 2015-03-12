"""
# Copyright (c) 1996, 1997, The Regents of the University of California.
# All rights reserved.  See Legal.htm for full text and disclaimer.

Pickle basic writer class PR by David Grote, LLNL
Heavily modified from PR.py originally written by Paul Dubois, LLNL, to use
pickle files.

This provides a nice wrapper over pickles for reading general
self-describing data sets written out with PWpickle.PW.

"""
__all__ = ['PR']
_version = '0.4'

import cPickle


class PRError(Exception):
    pass


class PR:
    """Pickle file read-access class.
Access to data in the file is available via attributes. For example, to get
the value of x from a file, do the following...

ff = PR(filename)
x = ff.x

There is also the read method
x = ff.read('x')
which is useful for names that are not usable as python attributes.
    """
    file_type = 'pickle'

    def __init__(self, filename):
        'PR(filename) opens file and reads in the pickled dictionary'
        self._filename = filename
        with open(filename, 'rb') as ff:
            self._pickledict = cPickle.load(ff)

    def __getattr__(self, name):
        return self._pickledict[name]

    def read(self, name):
        'read(name) = the value of name as a Python object.'
        return getattr(self, name)

    def close(self):
        'close(): close the file.'
        self._pickledict = {}

    def inquire_names(self):
        'inquire_names() = sorted list of names in the file'
        ls = list(self._pickledict.keys())
        ls.sort()
        return ls
