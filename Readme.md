# Certifair
Certifair is a framework for verifiying individual fairness properties globally for Neural Networks.

## System requirements
We tested Certifair on Ubuntu 20.04.4 LTS. We recommend having a GPU for training Neural Networks. It's also recommended to have at least 8 CPU cores for fairness verification.

## Installation
We recommend using `conda` for installing Certifair and we'll provide a step by step guide for the setup of conda environment on a Ubuntu.

Install the latest version of [Anaconda](https://docs.anaconda.com/anaconda/install/).

Create a conda environment with all the dependencies by running

`conda env create --name envname --file=environment.yml`

make sure to replace `envname` with the environment name you'd like and that the environment was created without errors.

## License 
PeregriNN -the underlying verifier- relies on Gurobi commercial solver which isn't open source. However, they provide a free academic license. Please request an academic license from [here](https://www.gurobi.com/academia/academic-program-and-licenses/)

## Test run
After installing Certifair and acquiring Gurboi license, we can test the installation by running

`python certifair.py german property1 german_p1_base --lr 0.007 --fr 0.0 --no-bound_loss --layers "30,30,1"`

## Reproduce Neurips results

You can reproduce the main results in table 2 by running 

`./scripts/tab2.sh`

Similarly, table 3 results can be reproduced by running 

`./scripts/tab3.sh`

## Help

For description of all args, run

`python certifair.py -h`


