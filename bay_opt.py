"""
Bayesian optimization loop for benchmarking existing molecule datasets

Author(s):
    Christina Schenk
    Fanjin Wang
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

from tqdm import tqdm

######################
###Define setting#####
######################

num_iter = 10
num_trial = 10
featurizer_name = 'rdkit'
partition_ratio = 0.05 # ratio of data to be used as starting set
# results
bests_over_trials = []

for trial in range(1,num_trial+1):
    
    print('Trial: ', trial)

    #############################
    ###Load and preprocess data##
    #############################
    # Load from pre-featurized data
    X, y = load_lipo_feat(filename='data/lipo_{}.csv'.format(featurizer_name))

    # Split data into start training and candidate sets
    X_train, X_candidate, y_train, y_candidate = train_test_split(
        X, y,
        test_size=1-partition_ratio,
        random_state=trial, #set random state for reproducibility, but vary in each trial
        shuffle=True
    )
    ###################################
    #####Train Surrogates##############
    ###################################
    # initialize surrogate
    my_surrogate = GPRQSurrogate()
    my_surrogate.load_data(train_x=X_train, train_y=y_train)
    best_observed = y_train.max()

    #initialize the containers of new points and best observed values
    X_new_candidates , y_new_candidates = [],[]
    current_bests = []

    for iter in tqdm(range(1,num_iter+1)):

        # Fit surrogate model.
        my_surrogate.fit()
        # #!!! return hps

        ######################################################################
        #####Eval element in candidate set and max Acquisition function#######
        ######################################################################

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


    #save best_over_trials to csv after iteration
    bests_over_trials.append(current_bests)

#################################
########Save necessary data######
#################################
np.savetxt(f'results/lipo_{featurizer_name}_best_observed.csv', bests_over_trials, delimiter=',')

#bestx, besty, hps, iter, bestobservedylabel