"""
Bayesian optimization loop for benchmarking existing molecule datasets

Author(s):
    Christina Schenk

Created:
    03/26/24
"""

#Python packages:
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from surrogates import Surrogate
from surrogates import RandomForestSurrogate
from surrogates import GPTanimotoSurrogate
from surrogates import GPRQSurrogate

import numpy as np
from data_helper import gen_data_feat,load_lipo_feat
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#############################
###Load and preprocess data##
#############################

featurizer_name = 'rdkit'
partition_ratio = 0.2

# Load from pre-featurized data
X, y = load_lipo_feat(filename='data/lipo_{}.csv'.format(featurizer_name))

# Split data into start training and candidate sets
X_train, X_candidate, y_train, y_candidate = train_test_split(
    X, y,
    test_size=1-partition_ratio,
    random_state=1,
    shuffle=True
)
X_train_tens = torch.from_numpy(X_train)
X_candidate_tens = torch.from_numpy(X_candidate)
print(np.shape(X_train))
print(np.shape(X_candidate))
y_train_tens = torch.from_numpy(y_train).unsqueeze(-1)
######################
###Define setting#####
######################
lb = [0.0]*np.shape(X_train)[1] #what should be the bounds???
ub = [1.0]*np.shape(X_train)[1]
niter = 2#500
ntrial = 1#100

#example https://botorch.org/tutorials/compare_mc_analytic_acquisition
neg_hartmann6 = Hartmann(dim=6, negate=True)
my_surrogate = GPTanimotoSurrogate()#GPRQSurrogate()
my_surrogate.load_data(train_x=X_train, train_y=y_train)
#for k in range(1, ntrial+1):
for iter in range(1, niter+1):
    ###################################
    #####Train Surrogates##############
    ###################################
    #First, we generate some random data and fit a SingleTaskGP for a 6-dimensional synthetic test function 'Hartmann6'.
    #replace Hartmann by utility function that calls evaluates the points from the dataset
    #train_x = torch.rand(10, 6)
    #train_obj = neg_hartmann6(train_x).unsqueeze(-1)
    #print(train_obj.type())
    #print(np.shape(train_x))
    #model = SingleTaskGP(train_X=X_train_tens, train_Y=y_train_tens) # this is where our surrogate goes but format?
    #my_surrogate.fit()
    #!!! return hps
    mll = ExactMarginalLogLikelihood(my_surrogate.likelihood(X_candidate), my_surrogate)
    # Fit surrogate model.
    fit_gpytorch_mll(mll);

    ######################################################################
    #####Eval element in candidate set and max Acquisition function#######
    ######################################################################

    best_value = y.max()#train_obj.max()
    EI = ExpectedImprovement(model=my_surrogate, best_f=best_value)

    #Next, we optimize the analytic EI acquisition function using 50 random restarts chosen from 100 initial raw samples.
    new_point_analytic, _ = optimize_acqf(#need to check specifics of this function or write ourselves!
        acq_function=EI,
        bounds=torch.tensor([lb, ub]),#e.g. ub [1.0]*6
        q=1,#batchsize=1
        num_restarts=1,#n
        raw_samples=100,#c
        options={},
    )
    print(new_point_analytic)