# ppar

[![License: GPL-3.0](https://img.shields.io/github/license/TB-IKP/ppar?color=blue)](LICENSE)

ppar is a [python](https://www.python.org/) program facilitating the comparison 
of calculated parallel momentum distributions to experimental data.
Currently, data taken at [NSCL](https://nscl.msu.edu/)/[FRIB](https://frib.msu.edu/) at Michigan State University and 
the [RIBF](https://www.nishina.riken.jp/ribf/) at [RIKEN](https://www.riken.jp/en/) are supported.

## Dependencies

* [python](https://www.python.org/) (tested with 3.9, 3.10, and 3.11)
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [uproot](https://github.com/scikit-hep/uproot5)

For NSCL/FRIB data additionally

* [PyAtima](https://github.com/TB-IKP/PyAtima)
* [Atima](https://web-docs.gsi.de/~weick/atima/)

## Install

`ppar` can be obtained by cloning this repository to the local system and running

```
python3 setup.py install
```

in the command line.

## Usage

Minimal examples can be found in the [example](example) directory.
The data files can be requested from Tobias Beck (tobias.beck@kuleuven.be).