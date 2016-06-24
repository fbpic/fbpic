# Installation of FBPIC on the Titan supercomputer

## Overview

Titan is a supercomputer at the Oakridge Leadership Computing Facility (OLCF).

Installation and usage of FBPIC requires the following steps:
* Installation of Anaconda and FBPIC
* Allocation of ressources and starting runs

## Installation

### Installation of Anaconda

In order to download and install Anaconda and FBPIC, follow the steps below:

- Download Miniconda:
```
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
```

- Execute the installation script, and use `/ccs/proj/<project id>`
as an install directory, so that the installation is accessible to the
compute nodes.
```
bash miniconda.sh -b -p /ccs/proj/<project id>/miniconda2
```
where the bracketed text should be replaced by the values for your account.

- Add the following lines at the end of your .bashrc
```
export $PATH=/ccs/proj/<project id>/miniconda2/bin:$PATH
export $LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ccs/proj/<project id>/miniconda2/lib
```
where again the bracketed text should be replaced by the values for your account.

### Installation of FBPIC and related packages

- Clone the `fbpic` repository using git.

- `cd` into the top folder of `fbpic` and install the dependencies:  
```
conda install --file requirements.txt
```

- Install `pyfftw` (not in the standard Anaconda channels, and thus it
requires a special command):  
```
conda install -c conda-forge pyfftw
```

- Install the `accelerate` package in order to be able to run on GPUs
```
conda install cudatoolkit=7.0
conda install accelerate
```
(The `accelerate` package is not free, but there is a 30-day free trial period,
  which starts when the above command is entered. For further use beyond 30
  days, one option is to obtain an academic license, which is also free. To do
  so, please visit https://www.continuum.io/anaconda-academic-subscriptions-available.)

- Install `fbpic`
```
python setup.py install
```

## Allocation of ressources and starting runs

Each node consists of 1 Nvidia K20 device.

In order to create a new simulation, create a new directory in
`$MEMBERWORK/` and copy your input script there:
```
mkdir $MEMBERWORK/<project id>/<simulation name>
cp fbpic_script.py $MEMBERWORK/<project id>/<simulation name>
```

### Interactive jobs

In order to request an interactive job:
```
qsub -I -A <project id> -l nodes=1,walltime=00:30:00 -q debug
```
Once the job has started, switch to your simulation directory
```
cd $MEMBERWORK/<project id>/<simulation name>
```
Then use `aprun` to launch the job on a GPU (even for single-node job)
```
aprun -n 1 python <fbpic_script.py>
```

### Batch jobs

Create a file `submission_script` with contains the following text:
```
#!/bin/bash
#PBS -A <project id>
#PBS -l walltime=<your walltime>
#PBS -l nodes=<number of nodes>

cd  $MEMBERWORK/<project id>/<simulation name>

aprun -n <number of nodes> python fbpic_script.py
```
