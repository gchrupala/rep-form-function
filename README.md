This repository contains the code used in the paper "Representation of
linguistic form and function in recurrent neural networks", the version
published in _Computational Linguistics_. 

The structure is the following:

- `doc`: Latex sources and figures for the paper
- `models`: RNN models trained on MS COCO
- `data`: data files
- `src`: code for analysis
   - [src/omission.py](src/omission.py) Omission scores
   - [src/depparse.py](src/depparse.py) Dependency parse using Spacy
   - [src/mutual.py](src/mutual.py) Mutual information scores
   - [src/example.py](src/examples.py) Examples
   - [src/ridge.py](src/ridge.py) Ridge regression models
   - [src/analysis.R](src/analysis.R) R code genarating figures
   - [src/analysis_mi.R](src/analysis_mi.R) R code genarating figures
   
The models were trained using the following software: https://github.com/gchrupala/reimaginet/releases/tag/v0.4

