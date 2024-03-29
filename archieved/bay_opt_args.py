"""
Bayesian optimization loop for benchmarking existing molecule datasets

Author(s):
    Christina Schenk
    Fanjin Wang
    Ankur Gupta
Created:
    03/26/24
"""

#Python packages:
import torch
from surrogates import RandomForestSurrogate
from surrogates import GPTanimotoSurrogate
from surrogates import GPRQSurrogate
from surrogates import acqf_EI

import numpy as np
from data_helper import gen_data_feat,load_lipo_feat
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from datetime import datetime
from tqdm import tqdm
import argparse

def run_bay(featurizer_name='mol2vec', feature_pca=False, partition_ratio=0.05, num_trial=5, num_iter=5):
    # results
    bests_over_trials = []
    mol_added_over_trials = []
    mol_start_over_trials = []

    #maybe add graph_kernel!
    if featurizer_name == 'rdkit' or featurizer_name == 'mordred' or featurizer_name == 'mol2vec':
        Surrogates = ['GPRQ', 'RandomForest']
    elif featurizer_name == 'ecfp' or 'e3fp':
        Surrogates = ['GPTanimoto', 'RandomForest']

    for surrogate in Surrogates:
        if surrogate == 'GPRQ':
            my_surrogate = GPRQSurrogate()
        elif surrogate == 'GPTanimoto':
            my_surrogate = GPTanimotoSurrogate()
        elif surrogate == 'RandomForest':
            my_surrogate = RandomForestSurrogate()

        for trial in range(1,num_trial+1):
            print('Trial: ', trial)
            print('------------------------------')
            print('Surrogate', surrogate)

            #############################
            ###Load and preprocess data##
            #############################
            # Load from pre-featurized data
            X, y = load_lipo_feat(filename='data/lipo_{}.csv'.format(featurizer_name))
            # generate an index for the molecules
            mol_track = np.arange(X.shape[0])

            # Split data into start training and candidate sets
            X_train, X_candidate, y_train, y_candidate, mol_track_train, mol_track_candidate = train_test_split(
                X, y, mol_track,
                test_size=1-partition_ratio,
                random_state=trial, #set random state for reproducibility, but vary in each trial
                shuffle=True
            )
            #Check Shape of X_train:
            if trial==1:
                print("\nShape of X_train:", np.shape(X_train))

            # Apply PCA to reduce dimensionality (optional)

            if feature_pca:
                # Standardize input data if needed
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_candidate = scaler.transform(X_candidate)

                pca = PCA(n_components=feature_pca)
                X_train = pca.fit_transform(X_train)
                X_candidate = pca.transform(X_candidate)

            ###################################
            #####Train Surrogates##############
            ###################################
            # initialize surrogate
            my_surrogate.load_data(train_x=X_train, train_y=y_train)
            best_observed = y_train.max()

            #initialize the containers of new points and best observed values
            # X_new_candidates , y_new_candidates = [],[]
            current_bests = []
            mol_added = []

            for iter in tqdm(range(1,num_iter+1)):

                # Fit surrogate model.
                my_surrogate.fit()

                ######################################################################
                #####Eval element in candidate set and max Acquisition function#######
                ######################################################################

                means, uncertainties = my_surrogate.predict_means_and_stddevs(X_candidate)

                # Calculate the Expected Improvement
                ei = acqf_EI(means, uncertainties, best_observed)

                # Find the index with the highest Expected Improvement
                new_index_in_ei = np.argmax(ei)
                new_x = X_candidate[new_index_in_ei]
                new_y = y_candidate[new_index_in_ei]

                # Add the new point to the training set
                my_surrogate.add_data(new_x, new_y)

                # Remove the new point from the candidate set
                X_candidate = np.delete(X_candidate, new_index_in_ei, axis=0)
                y_candidate = np.delete(y_candidate, new_index_in_ei)

                # Update the best observed value
                if new_y > best_observed:
                    best_observed = new_y

                # Record the new point and best observed value at this iteration
                #X_new_candidates , y_new_candidates = np.append(X_new_candidates, new_x), np.append(y_new_candidates, new_y)
                mol_added = np.append(mol_added, mol_track_candidate[new_index_in_ei])
                current_bests = np.append(current_bests, best_observed)

            #save best_over_trials to csv after iteration
            bests_over_trials.append(current_bests)
            mol_added_over_trials.append(mol_added)
            mol_start_over_trials.append(mol_track_train)

        #################################
        ########Save necessary data######
        #################################

        results = {
            'bests_over_trials': np.array(bests_over_trials),
            'mol_added': mol_added_over_trials,
            'mol_start': mol_start_over_trials
        }

        current_datetime = datetime.now().strftime("%m-%d_%H-%M-%S")
        np.save(f'results/lipo_{featurizer_name}_ratio{partition_ratio}_iter{num_iter}_trial{num_trial}'+str(surrogate)+f'_{current_datetime}.npy', results)
        #bestx, besty, hps, iter, bestobservedylabel
        torch.save(my_surrogate, f'results/model_{featurizer_name}_ratio{partition_ratio}_iter{num_iter}_trial{num_trial}'+str(surrogate)+f'_{current_datetime}.pickle')
        #torch.load('results/model.pickle')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Gaussian Process model')
    parser.add_argument('--featurizer_name', type=str, default='mol2vec', help='Name of the feature')
    parser.add_argument('--feature_pca', type=bool, default=False, help='use PCA or not')
    parser.add_argument('--partition_ratio', type=float, default=0.05, help='Partition Ratio')
    parser.add_argument('--num_trial', type=int, default=5, help='Number of Trials')
    parser.add_argument('--num_iter', type=int, default=5, help='Number of Iterations')

    args = parser.parse_args()
    run_bay(featurizer_name=args.featurizer_name, feature_pca=args.feature_pca, partition_ratio=args.partition_ratio, num_trial=args.num_trial, num_iter=args.num_iter)

    