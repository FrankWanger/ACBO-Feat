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
from surrogates import acqf_EI

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
######################
###Define setting#####
######################

# lb = #[0.0]*6
# ub = #[1.0]*6
num_iter = 10
num_trial = 100


###################################
#####Train Surrogates##############
###################################
# initialize ours

my_surrogate = GPRQSurrogate()
my_surrogate.load_data(train_x=X_train, train_y=y_train)
best_observed = y_train.max()

#initialize the containers of new points and best observed values
X_new_candidates , y_new_candidates= [],[]
current_bests = []

for iter in range(1,num_iter+1):

    # Fit surrogate model.
    my_surrogate.fit()

    #First, we generate some random data and fit a SingleTaskGP for a 6-dimensional synthetic test function 'Hartmann6'.
    #replace Hartmann by utility function that calls evaluates the points from the dataset
    #train_x = torch.rand(10, 6)
    #train_obj = neg_hartmann6(train_x).unsqueeze(-1)

    # model = #SingleTaskGP(train_X=train_x, train_Y=train_obj) # this is where our surrogate goes.
    # #!!! return hps
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_mll(mll);

    ######################################################################
    #####Eval element in candidate set and max Acquisition function#######
    ######################################################################

    # best_value = y.max()#train_obj.max()
    # EI = ExpectedImprovement(model=model, best_f=best_value)

    # #Next, we optimize the analytic EI acquisition function using 50 random restarts chosen from 100 initial raw samples.
    # new_point_analytic, _ = optimize_acqf(#need to check specifics of this function or write ourselves!
    #     acq_function=EI,
    #     bounds=torch.tensor([lb, ub]),#e.g. ub [1.0]*6
    #     q=1,#batchsize=1
    #     num_restarts=100,#n
    #     raw_samples=100,#c
    #     options={},
    # )
    # print(new_point_analytic)
    means, uncertainties = my_surrogate.predict_means_and_stddevs(X_candidate)
        
    # Calculate the Expected Improvement
    ei = acqf_EI(means, uncertainties, best_observed)

    # Find the index with the highest Expected Improvement
    new_index = np.argmax(ei)
    new_x = X_candidate[new_index]
    new_y = y_candidate[new_index]

    # Add the new point to the training set
    my_surrogate.add_data(new_x, new_y)

    # Remove the new point from the candidate set
    X_candidate = np.delete(X_candidate, new_index, axis=0)
    y_candidate = np.delete(y_candidate, new_index)

    # Update the best observed value
    if new_y > best_observed:
        best_observed = new_y

    # Record the new point and best observed value at this iteration
    X_new_candidates , y_new_candidates = np.append(X_new_candidates, new_x), np.append(y_new_candidates, new_y)
    current_bests = np.append(current_bests, best_observed)
    