# iontools

This is a Python module for dealing with the time evolution of single ions
interacting with a single sideband transition.  All the way through, we assume
that off-resonant transitions are negligible, so the time evolution is
completely analytical.

## Installation

The repository itself is a Python package, which depends on `numpy` (easily
available through any package manager) and [`qutip`](http://www.qutip.org/).
The latter is available through `conda` or `pip`, though further instructions
will be found on their website.

In any documentation, this module is referred to as `iontools`, so when cloning,
the command should look something like

```bash
git clone https://www.github.com/jakelishman/phd-iontools.git iontools
```

You can add the containing folder to the `PYTHONPATH`, or for quick things,
simply make sure a copy of the `iontools` folder is available in the directory
that the Python interpreter is being run from.
