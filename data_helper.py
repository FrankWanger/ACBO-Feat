import numpy as np
import os
import sys
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
    Ankur Gupta

Created: 
    03/25/24
'''

def gen_data_feat(featurizer_name='rdkit', raw_feature=False, save_data=False, save_filename='data/lipo_rdkit.csv', dataset='lipo'):
    '''
    Generate the feature matrix and labels for the dataset. Requires DeepChem/mol2vec/mordred/e3fp to be installed.
    See readme file for more details.

    dataset: str
        Name of the dataset to use. Currently limited to 'lipo'.

    featurizer_name: str
        Name of the featurizer to use. Options are 'rdkit', 'ecfp', 'mol2vec', 'mordred', 'e3fp'.

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
    from e3fp.fingerprint.generate import fprints_dict_from_mol
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Instantiate featurizers
    if featurizer_name not in ['rdkit', 'ecfp', 'mol2vec', 'mordred', 'e3fp']:
        raise ValueError('Invalid featurizer name. Options are rdkit, ecfp, mol2vec, mordred, e3fp.')
    else:
        featurizer = {
            'rdkit': RDKitDescriptors(),
            'ecfp': CircularFingerprint(),
            'mol2vec': Mol2VecFingerprint(),
            'mordred': MordredDescriptors(),
            'e3fp': RDKitDescriptors()  # We'll handle e3fp separately
        }[featurizer_name]

    # Check if the file already exists
    if os.path.isfile(save_filename):
        raise FileExistsError('File already exists. Please load from pre-featurized data or provide a different filename.')

    # Load the Lipophilicity dataset
    if dataset == 'lipo':
        tasks, datasets, transformers = dc.molnet.load_lipo(featurizer)

    # Load the original partition
    train_dataset, valid_dataset, test_dataset = datasets
    
    if featurizer_name == 'e3fp':
        
        # Generate e3fp fingerprints
        def generate_e3fp(smiles, bits=2048):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                if not mol.HasProp("_Name"):
                    mol.SetProp("_Name", smiles)
                try:
                    fingerprint = list(fprints_dict_from_mol(mol, bits=bits).values())[0][0]
                    # Create a flattened fingerprint vector of length 2048
                    flattened_fingerprint = np.zeros(bits, dtype=np.int8)
                    flattened_fingerprint[fingerprint.indices] = 1
                    #print(len(flattened_fingerprint), flush=True)
                    return flattened_fingerprint
                except Exception as e:
                    print(f"Error generating fingerprints for {smiles}: {str(e)}", flush=True)
                    return np.zeros(bits)
            return np.zeros(bits)

        X_train = np.array([generate_e3fp(smiles) for smiles in train_dataset.ids])
        X_valid = np.array([generate_e3fp(smiles) for smiles in valid_dataset.ids])
        X_test = np.array([generate_e3fp(smiles) for smiles in test_dataset.ids])
        X = np.vstack((X_train, X_valid, X_test))
        y = np.vstack((train_dataset.y, valid_dataset.y, test_dataset.y))
    else:
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

    return X, np.ravel(y)

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
    featurizer_name='e3fp'
    save_filename = os.path.join('data', 'lipo_{}.csv'.format(featurizer_name))

    # Generate and save the feature matrix and labels
    X, y = gen_data_feat(featurizer_name=featurizer_name, save_data=True, save_filename=save_filename)

    # or, load the pre-featurized data
    # X, y = load_lipo_feat(filename=save_filename)

    # Verify the shape of the feature matrix and labels
    print(X.shape, y.shape)
