# Contributing to FBPIC

## How to contribute

### Forking the repository

In order to contribute, please fork the [main repository](https://github.com/fbpic/fbpic):

- Click 'Fork' on the page of the main repository, in order to create a personal copy of this repository on your Github account.

- Clone this copy to your local machine:
```
git clone git@github.com:<YourUserLogin>/fbpic.git
```

### Implementing a new feature and adding it to the main repository

- Switch to the development branch
```
git checkout dev
```
and install it
```
python setup.py install
```

- Start a new branch from the development branch, in order to
implement a new feature. (Choose a branch name that is representative of the
feature that you are implementing, e.g. `add-quadratic-deposition` or
`fix-matplotlib-errors`)
```
git checkout -b <NewBranchName>
```

- Start coding. When your changes are ready, commit them.
```
git add <ChangedFiles>
git commit
```

- Synchronize your branch with the main repository. (It may have
  changed while you where implementing local changes.) Resolve merging
  issues if any, and commit the corresponding changes.
```
git pull git@github.com:fbpic/fbpic.git dev
```

- Test and check your code:
  - Use [pyflakes](https://pypi.python.org/pypi/pyflakes) and
[pep8](https://pypi.python.org/pypi/pep8) to detect any potential bug.
  ```
  cd fbpic/
  pyflakes .
  ```
  - Make sure that the tests pass (please install `openPMD-viewer` first)
  ```
  python setup.py install
  pip install matplotlib openPMD-viewer
  python setup.py test
  ```
  (Be patient: the tests can take approx. 5 min.)

- Push the changes to your personal copy on Github
```
git push -u origin <NewBranchName>
```

- Go on your Github account and create a pull request between **your
  new feature branch** and the **dev branch of the main
  repository**. Please add some text to the pull request to describe
  what feature you just implemented and why. Please also make sure that
  the automated tests (on Github) return no error.

## Style and conventions

- Document the functions and classes that you write, by using a
  [docstring](https://www.python.org/dev/peps/pep-0257/). List the
  parameters in and describe what the functions return, according to
  [Numpy style](https://github.com/numpy/numpy/blob/main/doc/HOWTO_DOCUMENT.rst.txt),  as in this example:

```python
def print_simulation_setup( comm, use_cuda ):
    """
    Print message about the number of proc and
    whether it is using GPU or CPU.

    Parameters
    ----------
    comm: an fbpic BoundaryCommunicator object
        Contains the information on the MPI decomposition

    use_cuda: bool
        Whether the simulation is set up to use CUDA
    """
```
Don't use documenting styles like `:param:`, `:return:`, or
`@param`, `@return`, as they are less readable.


- Lines of code should **never** have [more than 79 characters per line](https://www.python.org/dev/peps/pep-0008/#maximum-line-length).

- Names of variables, functions should be lower case (with underscore
  if needed: e.g. `get_filter_array`). Names for classes should use the
  CapWords convention (e.g. `Fields`). See [this page](https://www.python.org/dev/peps/pep-0008/#prescriptive-naming-conventions) for more details.
