# AC BO Hackathon 2024
This is a repo created for the 2024 AC BO Hackathon.

## Project
BO for Drug Discovery: What is the role of molecular representation?

## Prerequisites
To replicate the work, the following dependancies are necessary:
- python 3.9
- botorch (framework for BO)
- deepchem (framework for cheminformatics)
- mol2vec (featurization)
- mordred (featurization)

To set up the environment, follow the steps:
```bash
conda create --name bofeat python=3.9
conda activate bofeat
pip install botorch deepchem
pip install git+https://github.com/samoturk/mol2vec
```
P.S. Installation with conda manager is not recommended as it might led to incompatibility issue

P.P.S. Make sure pip is from the newly created created `bofeat` environment. If you're using a Unix-based OS, you can use which pip to check