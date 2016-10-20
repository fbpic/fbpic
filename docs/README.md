This directory contains the source files for the documentation of FBPIC.

The documentation is written using [Sphinx](http://www.sphinx-doc.org/en/stable/). The source files are located in `source/` and are written in [reStructuredText format] (http://www.sphinx-doc.org/en/stable/rest.html#rst-primer).

In order to generate the documentation in html format, first make sure
that you have Sphinx installed (`pip install Sphinx sphinx_rtd_theme`), and that FBPIC
is properly installed. Then type:
```
make html
```

To view the documentation, enter the directory `build/html` and open
`index.html` with a web browser.
