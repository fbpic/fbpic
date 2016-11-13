# Creating a new release

This document is only relevant for maintainers of fbpic. It
explains how to create a new release. In future versions of this
packages, some of the steps below will be automatized.

## Preparing your environment for a release

Make sure that your local environment is ready for a full release on
PyPI and conda. In particular:

- you should install the packages
[`pypandoc`](https://pypi.python.org/pypi/pypandoc/),
[`twine`](https://pypi.python.org/pypi/twine),
[`conda-build`](http://conda.pydata.org/docs/commands/build/conda-build.html)
and [`anaconda-client`](https://anaconda.org/anaconda/anaconda-client)
- you should have a registered account on [PyPI](https://pypi.python.org/pypi) and [test PyPI](https://testpypi.python.org/pypi), and your `$HOME` should contain a file `.pypirc` which contains the following text:
  ```
[distutils]
index-servers=
	pypitest
	pypi

[pypitest]
repository = https://testpypi.python.org/pypi
username = <yourPypiUsername>

[pypi]
repository = https://pypi.python.org/pypi
username = <yourPypiUsername>
```
- you should have a registered account on [Anaconda.org](https://anaconda.org/)

## Creating a release on Github

- Make sure that the version number in `fbpic/__init__.py`
  corresponds to the new release, and that the corresponding changes have been
  documented in `CHANGELOG.md`.

- If everything works fine, then merge the `dev` version into `master`
and upload it to Github:
```
git checkout master
git merge dev
git push
```

- Create a new release through the graphical interface on Github

## Uploading the package to PyPI

- Upload the package to [PyPI](https://pypi.python.org/pypi):
```
rm -rf dist
python setup.py sdist bdist_wheel
twine upload dist/* -r pypi
```
(NB: You can also first test this by uploading the package to
[test PyPI](https://testpypi.python.org/pypi) ; to do so, simply
replace `pypi` by `pypitest` in the above set of commands)



