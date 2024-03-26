import numpy as np
import os

'''
Summary:
    data_helper.py is a helper script that generates or load
    feature matrices and labels for the Lipophilicity dataset. 

    gen_data_feat funciton requires the DeepChem, mol2vec, mordred 
    library to be installed. To avoid incompatibility issues, it is 
    highly recommended to use pre-featurized data. Mordred 
    fingerprint takes ~10 minutes to generate featurize
    other fingerprints take ~1 minute to generate

    load_lipo_feat function loads the pre-featurized data.

Author(s):
    Fanjin Wang

Created: 
    03/25/24
'''

def gen_data_feat(featurizer_name='rdkit', raw_feature=False, save_data=False, save_filename='data/lipo_rdkit.csv',dataset = 'lipo'):
    '''
    Generate the feature matrix and labels for the dataset. Requires DeepChem/mol2vec/mordred to be installed. 
    See readme file for more details.

    dataset: str
        Name of the dataset to use. Currently limited to 'lipo'.

    featurizer_name: str
        Name of the featurizer to use. Options are 'rdkit', 'ecfp', 'mol2vec', 'mordred'.

    raw_feature: bool
        If True, keep NaN values in the feature matrix. If False, remove NaN values in mordred and rdkit featurizers.

    Returns:
        X: np.ndarray
            Feature matrix of shape (N, D) where N is the number of samples and D is the number of features.
        y: np.ndarray
            Labels of shape (N,).
    '''
    import deepchem as dc
    from deepchem.feat import CircularFingerprint, RDKitDescriptors, Mol2VecFingerprint, MordredDescriptors

    # Instantiate featurizers
    if featurizer_name not in ['rdkit', 'ecfp', 'mol2vec', 'mordred']:
        raise ValueError('Invalid featurizer name. Options are rdkit, ecfp, mol2vec, mordred.')
    else:
        featurizer = {
            'rdkit': RDKitDescriptors(),
            'ecfp': CircularFingerprint(),
            'mol2vec': Mol2VecFingerprint(),
            'mordred': MordredDescriptors()
        }[featurizer_name]

    # Check if the file already exists
    if os.path.isfile(save_filename):
        raise FileExistsError('File already exists. Please load from pre-featurized data or provide a different filename.')
    
    # Load the Lipophilicity dataset
    if dataset == 'lipo':
        tasks, datasets, transformers = dc.molnet.load_lipo(featurizer)

    # Load the original partition
    train_dataset, valid_dataset, test_dataset = datasets

    # Get the feature matrix and labels 
    X, y = np.vstack((train_dataset.X,valid_dataset.X,test_dataset.X)),np.vstack((train_dataset.y,valid_dataset.y,test_dataset.y))
    
    # Remove NaN values from the feature matrix
    if not raw_feature:
        # Find the columns in your data that have NaN values
        nan_cols = np.any(np.isnan(X), axis=0)

        # Select only the columns in your data that don't have NaN values
        X = X[:, ~nan_cols]

    if save_data:
        np.savetxt(save_filename, np.hstack((X,y)), delimiter=',')

    return  X,np.ravel(y)

def load_lipo_feat(filename='./data/lipo_rdkit.csv'):
    '''
    Load the feature matrix and labels for the Lipophilicity dataset.

    Returns:
        X: np.ndarray
            Feature matrix of shape (N, D) where N is the number of samples and D is the number of features.
        y: np.ndarray
            Labels of shape (N,).
    '''
    # Load the data
    if os.path.isfile(filename):
        data = np.loadtxt(filename, delimiter=',')
        X, y = data[:, :-1], data[:, -1] # Last column is the label
    else:
        raise FileNotFoundError('Pre-featurized data not found. Please use gen_data_feat instead.')
    return X, y

if __name__ == '__main__':

    # Example usage
    featurizer_name='rdkit'
    save_filename = os.path.join('data', 'lipo_{}.csv'.format(featurizer_name))

    # Generate and save the feature matrix and labels
    X, y = gen_data_feat(featurizer_name=featurizer_name, save_data=True, save_filename=save_filename)

    # or, load the pre-featurized data
    # X, y = load_lipo_feat(filename=save_filename)

    # Verify the shape of the feature matrix and labels
    print(X.shape, y.shape)