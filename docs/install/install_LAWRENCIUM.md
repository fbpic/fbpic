# Installation of FBPIC on Lawrencium

## Overview

Lawrencium is a local cluster at Lawrence Berkeley National lab.
It has two NVIDIA K20 GPUs.

## Connecting to Lawrencium

Lawrencium uses a one-time password (OTP) system. Before being able to connect to Lawrencium via ssh, you need to configure an OTP Token, using [these instructions](https://commons.lbl.gov/display/itfaq/Installing+and+Configuring+the+OTP+Token).

Once your OTP token is configured, you can connect by using
```
ssh <username>@lrc-login.lbl.gov
```
When prompted for the password, generate a new one-time password with the Pledge application, and enter it at the prompt.

## Installation 

### Installation of Anaconda

In order to download and install Anaconda and FBPIC, follow the steps below:

- Download Anaconda:
```
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.1-Linux-x86_64.sh
```

- Execute the Anaconda installation script
```
bash Anaconda2-2.4.1-Linux-x86_64.sh
```
When the installer suggests to add the `anaconda` path in your `.bashrc`, please answer `yes`.

- Add the following lines at the end of your .bashrc
```
module load glib/2.32.4 
module load cuda
```

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
conda install accelerate
conda install accelerate_cudalib
```
(The `accelerate` package is not free, but there is a 30-day free trial period,
  which starts when the above command is entered. For further use beyond 30
  days, one option is to obtain an academic license, which is also free. To do
  so, please visit [this link](https://www.continuum.io/anaconda-academic-subscriptions-available).)

- Install `fbpic`
```
python setup.py install
```

## Running simulations

### Preparing a new simulation

It is adviced to use the directory `/global/scratch/<yourUsername>`
for faster I/O access, where `<yourUsername>` should be replaced by
your username.

In order to prepare a new simulation, create a new subdirectory within
the above-mentioned directory, and copy your input script there.

### Interactive jobs

In order to request a node with a GPU:
```
salloc --time=00:30:00 --nodes=1 --partition lr_manycore  --constraint=lr_kepler --account=<yourAccountName> --qos=lr_normal
```
where `<yourAccountName>` should be replace by the account that will
be charged for the simulation.

Once the job has started, type
```
srun --pty -u bash -i
```
in order to connect to the node that has been allocated. Then `cd` to
the directory where you prepared your input script and type
```
python <fbpic_script.py>
```

### Batch job

Create a new file named `submission_file` in the same directory as your
input script (typically this directory is a subdirectory of
`/global/scratch/<yourUsername>`). Within this new file, copy the
following text (and replace the bracketed text by the proper values).
```
#!/bin/bash
#SBATCH -J my_job
#SBATCH --partition=lr_manycore
#SBATCH --constraint lr_kepler
#SBATCH --time <requestedTime>
#SBATCH --nodes 1
#SBATCH --account <yourAccountName>
#SBATCH --qos lr_normal
#SBATCH -e my_job.%j.err
#SBATCH -o my_job.%j.out

python <fbpic_script.py>
```
Then run:
```
sbatch submission_file
```
In order to see the queue:
```
squeue -p lr_manycore
```

### Transfering data to your local computer for visualization/post-processing

In order to transfer your data to your local machine, you need to
connect to the transfer node. From a Lawrencium login node, type:
```
ssh lrc-xfer.scs00
```
You can then use for instance `rsync` to transfer data to your local computer.

