Installation of FBPIC  on the JURECA SUPERCOMPUTER
=======================================

Overview
-------

JURECA is a supercomputer at the Forschungszentrum Juelich.

Installation and usage of FBPIC requires the following steps:

* Loading the cluster modules
* Installation of Anaconda and additional packages
* Installation of the package `mpi4py`
* Installation of FBPIC
* Allocation of ressources and starting runs

Loading the cluster modules
-------------------

On the JURECA cluster, the correct modules to use a fast CUDA-aware MPI distribution need to be loaded.

Therefore, the `.bashrc` should contain the following:

```
module use /usr/local/software/jureca/OtherStages
module load Stages/Devel
module load gmv2olfc
```

Installation of Anaconda and additional packages
-------------------

It is recommended to use Anaconda on the cluster.

Download and installation:

```
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.1-Linux-x86_64.sh

bash Anaconda2-2.4.1-Linux-x86_64.sh
```

Then additional packages need to be installed:
`
conda install numba
`
`
conda install accelerate
`
`
conda install accelerate-cudalib
`
`
conda install matplotlib
`
`
conda install h5py
`

pyFFTW needs to be installed from a third-party source:
`
conda install -c https://conda.anaconda.org/mforbes pyfftw
`
This command typically downgrades the version of numpy, so please type
afterwards
`
conda upgrade numpy
`

It is important that the following packages are **NOT** installed directly with Anaconda: `mpich` and `mpi4py`

Installation of the package `mpi4py`
-------------------

The package `mpi4py` needs to be build with the cluster MPI distribution. Therefore, after loading the correct modules, it can be installed by:

```
pip install mpi4py
```

One can check if the correct MPI is linked by opening a `python` shell and checking:

```
from mpi4py import MPI
MPI.Get_library_version()
```

Installation of FBPIC
-------------------

Finally, FBPIC can be installed as usual with `python setup.py install`.

Allocation of ressources and starting runs
-------------------

In the following, it is explained how to allocate and use **interactive** jobs on JURECA. For the usage of normal jobs, one can use the similar commands in a jobscript. More information can be found here:

`
http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/UserInfo/UserInfo_node.html
`

**Allocation of ressources**

**CPU:**
CPU nodes consist of 24 cores. Allocation of two nodes for 60 minutes:

`
salloc --nodes=2 --time=00:60:00
`

**GPU:**
GPU nodes consist of 2 Nvidia K80 Devices, i.e. 4 GPUs. Allocation of 4 GPUs (2 nodes) for 60 minutes:

`
salloc --nodes=2 --partition=gpus --time=00:60:00 --gres=gpu:4
`

**Starting an interactive run**

The following command starts an interactive run (run_file.py) with 8 tasks (e.g. 8 GPUs). `--pty` activates continuous console output and `--forward-x`enables X-forwarding if the connection to JURECA was established with `ssh -Y username@jureca.fz-juelich.de`.

`
srun --ntasks=8 --forward-x --pty python run_file.py
`

