import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_helper import load_lipo_feat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from surrogates import Surrogate
from surrogates import RandomForestSurrogate
from surrogates import GPTanimotoSurrogate
from surrogates import GPRQSurrogate

# Load toy dataset for testing.
diabetes = load_diabetes(scaled=False)

# Load the Lipophilicity dataset.
X, y = load_lipo_feat(filename='./data/lipo_rdkit.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    shuffle=True
)

# Define surrogate model.
my_surrogate = GPRQSurrogate()
my_surrogate.load_data(train_x=X_train, train_y=y_train)

# Fit surrogate model.
my_surrogate.fit()

# Get means and uncertainties from surrogate model.
means, uncertainties = my_surrogate.predict_means_and_stddevs(X_test)
print(f'Test shape: {X_test.shape}')
print(f'Mean shape: {means.shape}')
print(f'Uncertainty shape: {uncertainties.shape}')

# Report results of model fit.
print(f'R^2 Score on test set: {r2_score(y_test, means)}')