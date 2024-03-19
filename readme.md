# AC BO Hackathon 2024
This is a repo created for the 2024 AC BO Hackathon.

## Project
BO for Drug Discovery: What is the role of molecular representation?

## Prerequisites
To replicate the work, the following dependancies are necessary:
- botorch
- deepchem
- rdkit


```bash
conda create --name bofeat python=3.9
conda activate bofeat
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda install -c conda-forge rdkit deepchem
```
(Installation order cannot be swapped as it might led to incompatibility issue )