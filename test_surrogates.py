import argparse
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from surrogates import RandomForestSurrogate
from surrogates import GPTanimotoSurrogate
from surrogates import GPRQSurrogate
from data_helper import load_lipo_feat

# Get user input for what model to test.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['rdkit', 'mol2vec'], default='rdkit')
parser.add_argument('--model', choices=['RF', 'Tanimoto', 'RQ'], default='RF')
args = parser.parse_args()

if args.model == 'RF':

    # Load toy dataset for testing.
    X, y = load_lipo_feat(filename=f'./data/lipo_{args.dataset}.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.2,
        random_state=1,
        shuffle=True
    )

    # Define random forest surrogate.
    my_surrogate = RandomForestSurrogate()

elif args.model == 'Tanimoto':

    # Load toy dataset for binary fingerprint testing.
    dataset = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv')
    features = dataset['smiles']
    mols = [Chem.MolFromSmiles(smi) for smi in features]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=1024) for mol in mols]
    X = np.array(fps).astype(np.int32)
    y = dataset['expt'].to_numpy().astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        shuffle=True
    )

    # Define Tanimoto GP surrogate.
    my_surrogate = GPTanimotoSurrogate()

elif args.model == 'RQ':

    # Load lipo dataset for testing.
    X, y = load_lipo_feat(filename=f'./data/lipo_{args.dataset}.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        shuffle=True
    )

    # Define GPRQ surrogate.
    my_surrogate = GPRQSurrogate()

else:
    raise Exception('No model specified!')

# Load training data to surrogate model.
my_surrogate.load_data(train_x=X_train, train_y=y_train)

# Fit surrogate model.
my_surrogate.fit(progress=True)

# Get means and uncertainties from surrogate model.
means, uncertainties = my_surrogate.predict_means_and_stddevs(X_test)
print(f'Test shape: {X_test.shape}')
print(f'Mean shape: {means.shape}')
print(f'Uncertainty shape: {uncertainties.shape}')

# Report results of model fit.
print(f'R^2 Score on test set: {r2_score(y_test, means)}')