Installation of numba on MacOSX
======================

Overview
-------

Due to an incompatibility between llvmlite-0.3.0 and numba-0.17.0, installation through macports will fail. This file explains how to install numba in this case.

If you are using numba-0.18 or a more recent version, the installation note below is irrelevant. Please visit [the numba installation page](https://pypi.python.org/pypi/numba/).

Installation
---------

Go through the following step :

* Uninstall any previous version of numba and llvmlite in macports and
  pip
  
`sudo port uninstall py27-numba`

`sudo port uninstall py27-llvmlite`

`sudo pip uninstall numba`

`sudo pip uninstall llvmlite`

* Install llvm-3.5 via macports

`sudo port install llvm-3.5`

* Update the location database and find the location of the `llvm-config` executable

`sudo /usr/libexec/locate.updatedb`

`locate llvm-config`

* Export that location as an environnement variable and add it to the sudoer environnement variables

`export LLVM_CONFIG=/path/to/llvm-config`

`sudo visudo`: `Defaults        env_keep += "LLVM_CONFIG"`

* Install llvmlite with pip (this install llvmlite-0.2.2 instead of the faulty llvmlite-0.3.0)

`sudo pip install llvmlite`

* Install numba with pip

`sudo pip install numba`
