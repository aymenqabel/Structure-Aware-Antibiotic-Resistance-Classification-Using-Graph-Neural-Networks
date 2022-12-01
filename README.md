# Structure-Aware-Antibiotic-Resistance-Classification-Using-Graph-Neural-Networks
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is an implementation of ["Structure-Aware-Antibiotic-Resistance-Classification-Using-Graph-Neural-Networks"](https://openreview.net/pdf?id=_BjtIlib8N9)
![](./figures/architecture.pdf)
 

## Abstract
Antibiotics are traditionally used to treat bacterial infections. However, bacteria can develop immunity to drugs, rendering them ineffective and thus posing a serious threat to global health. Identifying and classifying the genes responsible for this resistance is critical for the prevention, diagnosis, and treatment of infections as well as the understanding of its mechanisms. Previous methods developed for this purpose have mostly been sequence-based, relying on the comparison to existing databases or machine learning models trained on sequence features. However, genes with comparable functions may not always have similar sequences. Consequently, in this paper, we develop a deep learning model that uses the protein structure as a complement to the sequence to classify novel Antibiotic Resistant Genes (ARGs), which we expect to provide more useful information than the sequence alone. The proposed approach consists of two steps. First, we capitalize on the much-celebrated AlphaFold model to predict the 3D structure of a protein from its amino acid sequence. Then, we process the sequence using a transformer-based protein language model and apply a graph neural network to the graph extracted from the structure. We evaluate the proposed architecture on a standard benchmark dataset on which we find it to outperform state-of-the-art methods.

## Requirements
An appropriate virtual environment can be created by `conda`  using the provided environments file,
```
conda env create -f environment.yml
```
## Usage

We provide the implementation of the ESM + GNN pipeline. The data will be available soon. 
```
python train.py 
```
## Contribution
- Aymen Qabel (qabel@lix.polytechnique.edu)
- Sofiane Ennadir
- Giannis Nikolentzos
- Johannes F. Lutzeyer
- Michail Chatzianastasis
- Henrik Boström
- Michalis Vazirgiannis 
