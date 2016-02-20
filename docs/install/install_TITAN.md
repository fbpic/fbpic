Installation of FBPIC on the Titan supercomputer
=======================================

Overview
-------

Titan is a supercomputer at the Oakridge Leadership Computing Facility (OLCF).

Installation and usage of FBPIC requires the following steps:
* Installation of Anaconda and FBPIC
* Allocation of ressources and starting runs

Installation of Anaconda
------------------------

In order to download and install Anaconda and FBPIC, follow the steps below:

- Download Anaconda:
```
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.1-Linux-x86_64.sh
```

- Get the name of your $MEMBERWORK directory and write it down
```
echo $MEMBERWORK
```

- Execute the Anaconda installation script
```
bash Anaconda2-2.4.1-Linux-x86_64.sh
```
**DO NOT use the default installation directory:** When prompted for an
installation directory, use
```
<$MEMBERWORK directory>/<project id>/anaconda2
```
where the bracketed text should be replaced by the values for your account.

- Add the following lines at the end of your .bashrc
```
export $PATH="<$MEMBERWORK directory>/<project id>/anaconda2/bin:$PATH"
export $LD_LIBRARY_PATH="<$MEMBERWORK directory>/<project id>/anaconda2/lib:$LD_LIBRARY_PATH"
```
where again the bracketed text should be replaced by the values for your account.

Installation of FBPIC and related packages
------------------------------------------

- Clone the `fbpic` repository using git.

- `cd` into the top folder of `fbpic` and install the dependencies:  
```
conda install --file requirements.txt
```

- Install `pyfftw` (not in the standard Anaconda channels, and thus it
requires a special command):  
```
conda install -c https://conda.anaconda.org/richli pyfftw
conda upgrade numpy
```
**Important:** Do not use the URL https://conda.anaconda.org/mforbes, since it
is known to cause bugs on the Titan cluster.

- Install the `accelerate` package in order to be able to run on GPUs
```
conda install accelerate
conda install accelerate_cudalib
```
(The `accelerate` package is not free, but there is a 30-day free trial period,
  which starts when the above command is entered. For further use beyond 30
  days, one option is to obtain an academic license, which is also free. To do
  so, please visit https://www.continuum.io/anaconda-academic-subscriptions-available.)

- Install `fbpic`
```
python setup.py install
```

Allocation of ressources and starting runs
-------------------
Each node consists of 1 Nvidia K20 device.

**Interactive jobs:**

In order to request an interactive job:
```
qsub -I -A <project id> -l nodes=1,walltime=00:30:00 -q debug
```
Once the job has started, switch to the `$MEMBERWORK/` directory and
copy your script there.
```
mkdir $MEMBERWORK/<project id>/<simulation name>
cp <fbpic_script.py> $MEMBERWORK/<project id>/<simulation name>
cd $MEMBERWORK/<project id>/<simulation name>
```
Then use `aprun` to launch the job on a GPU (even for single-node job)
```
aprun -n 1 python <fbpic_script.py>
```
