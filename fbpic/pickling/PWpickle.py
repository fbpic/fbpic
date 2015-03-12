"""
# Copyright (c) 2008, LLNS LLC
# All rights reserved.  See LEGAL.LLNL for full text and disclaimer.

Pickled based writer class PW by David Grote, LLNL
Heavily modified from PW.py originally written by Paul Dubois, LLNL, to use
pickle files.

This puts everything into a dictionary and when close is called, pickles the
dictionary into the file.

"""
__all__ = ['PW']
import cPickle
import warnings

_version = '0.4'

class PWError(Exception):
    pass

class PW:
    """Pickle file writing class.
Write data to the file using attributes. For example, to write
the value of x to a file, do the following...

ff = PW(filename)
ff.x = x

There is also the write method
x = ff.write('x')
which is useful for names that are not usable as python attributes.
Note that only things which can be pickled and unpickled are written out.
    """

    file_type = "pickle"

    def __init__(self, filename, mode='w', protocol=-1):
        "PW(filename='', mode='w') creates filename"
        self.__dict__['_filename'] = filename
        self.__dict__['_protocol'] = protocol
        assert mode in ['w','a'],Exception("Improper mode: " + mode)
        if mode == 'a':
            # --- If append, read in all data from the file.
            with open(filename,mode='rb') as ff:
                self.__dict__['_pickledict'] = cPickle.load(ff)
        else:
            self.__dict__['_pickledict'] = {}

        self.__dict__['_file'] = open(filename,mode='wb')

    def __del__(self):
        "Close any file open when this object disappears."
        self.close()

    def close(self):
        "close(): close the file."
        if self._file is not None:

            # --- Write out the dictionary of things to be pickled,
            # --- pickling it all at once.
            cPickle.dump(self._pickledict,self._file,self._protocol)

            self._file.close()

        # --- Reset various quantities
        self.__dict__['_file'] = None
        self.__dict__['_pickledict'] = {}

    def __setattr__(self, name, value):
        self.write(name, value)

    def __getattr__(self, name):
        return self._pickledict[name]

    def __repr__(self):
        if self.is_open():
            return 'PWpickle file %s is opened for writing.'%self._filename
        else:
            return PWError,'(PWpickle object not open on any file)'

    __str__ = __repr__

    def is_open(self):
        "is_open() = true if file is open"
        return self._file is not None

    def write(self, name, quantity, title='', lAcceptMainClass=0):
        """Write quantity to file as 'name'"""
        assert self.is_open(),'PWpickle object not open for write.'

        # --- Do some validation on the input, making sure that it can be
        # --- pickled and unpickled properly.
        try:
            try:
                if (quantity.__class__.__module__ == '__main__'
                    and not lAcceptMainClass):
                    warnings.warn('%s is being skipped since it is an instance of a class defined in main and therefore cannot be unpickled'%name)
                    return
            except:
                pass
            try:
                if (quantity.__module__ == '__main__' and not lAcceptMainClass):
                    warnings.warn('%s is being skipped since it is a class defined in main and therefore cannot be unpickled'%name)
                    return
            except:
                pass
            # --- Check if an object can be pickled. This will be inefficient
            # --- for large objects since they will be pickled twice, but this
            # --- is the only safe way.
            try:
                q = cPickle.dumps(quantity,self._protocol)
                del q
            except:
                warnings.warn('%s is being skipped since it could not be pickled'%name)
                return
            # --- The object can now be safely added to the dictionary of
            # --- things to be pickled.
            self._pickledict[name] = quantity
            return
        except:
            pass

        raise PWError,"Could not write the variable %s"%name

if __name__ == "__main__":
    f=PW("foo.pickle")
    a = 1
    b = 2.0
    c = "Hello world"
    import numpy
    d = numpy.array([1.,2., 3.])
    e = numpy.array([1,2,3])
    g = numpy.array(["hello", "world", "array"])
    h = numpy.array([[1.,2.,3.], [4.,5.,6]])
    k = 3
    f.a = a
    f.b = b
    f.c = c
    f.d = d
    f.e = e
    f.g = g
    f.h = h
    f.close()
    f=PW("foo.pickle",mode='a')
    f.k = k
    f.close()
# read-back test
    from PRpickle import PR
    f = PR('foo.pickle')
    for x in f.inquire_names():
        print x, "is", eval(x), ", in file it is", eval('f.'+x)
    f.close()
