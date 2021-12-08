# GDR REST Machine Learning Meeting: Hands On Session
Interactive session to implement deep learning methods in python using TensorFlow for the REST GDR Machine Learning Discussion Meeting.

## Dependencies:
- python = ">=3.7,<3.10"
- numpy = "^1.21.4"
- scipy = "^1.7.3"
- tensorflow = "^2.7.0"
- matplotlib = "^3.5.0"
- jupyter = "^1.0.0"
- jupyterlab = "^3.2.4"

## Installation
To install using pip:

`pip install -r requirements.txt`

To install using poetry:

`poetry install`

`poetry shell`

## Usage
To run the notebook in jupyter lab, simply use:

`jupyter lab`

Alternativly using jupyter notebook:

`jupyter notebook`

Or, the notebook can be run as a raw python script:

`python notebook.py`

## Directory structure
```
|-- data.py               -> Code used to generate data for sections 3 and 4.
|                           (For interest only, uses the iDEA code so cannot be run, public release coming soon!)
|-- notebook.ipynb        -> Session notebook.
|-- notebook.py           -> Session notebook as executable script.
|-- notebook.pdf          -> Printout of notebook.
|-- requirements.txt      -> Define project dependencies for pip.
|-- pyproject.toml        -> Define project dependencies for poetry.
|-- *.db                  -> Data files.
|                            (V -> external potential, density -> charge density, E -> total energy).
|                            All are pickles numpy arrays.
|-- *.png                 -> Various diagrams.
|-- *.pdf                 -> Results generated on running notebook.
```
