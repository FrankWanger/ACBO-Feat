# AC BO Hackathon 2024
This is a repo created for the [2024 AC BO Hackathon](https://ac-bo-hackathon.github.io/agenda/)
## Project
BO for Drug Discovery: What is the role of molecular representation?

Project leads
- Fanjin Wang (University College London)
Contributors
- Quinn Gallagher (Princeton University)
- Ankur Gupta (Lawrence Berkeley National Laboratory)
- Christina Schenk (IMDEA Materials Institute)

## Questions we are asking:
 - How fingerprints play a role in BO process
 - Will dimensionality reduction help with BO performance?
 - Instead of GP surrogate, will I get better results with a random forest surrogate?


## Poster
![Project 8](/figures/poster.png)

## Results
![Performance of Fingerprints](/figures/result1.png)

mol2vec performed the best in our benchmark!

## Prerequisites
To replicate the work, the following dependancies are necessary:
- python 3.9
- botorch (framework for BO)
- deepchem (framework for cheminformatics)
- mol2vec (featurization)
- numpy
- scikit-learn
- scikit-optimize
- torch
- gpytorch

To set up the environment, follow the steps:
```bash
conda create --name bofeat python=3.9
conda activate bofeat
pip install botorch deepchem numpy scikit-learn scikit-optimize torch gpytorch
pip install git+https://github.com/samoturk/mol2vec
```
P.S. Installation with conda manager is not recommended as it caused weird incompatibility issue.

P.P.S. Make sure pip is from the newly created `bofeat` environment. If you're using a Unix-based OS, you can use `which pip` to check
