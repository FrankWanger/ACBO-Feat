import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical
import torch
import gpytorch

'''
Summary:
    surrogates.py includes all surrogate models used in
    ACBO-FEAT. This includes a base surrogate model, 
    a random forest surrogate model, a GP surrogate model
    with a Tanimoto kernel for bit-vector features, and a
    GP surrogate model with an anisotropic RQ kernel for 
    continuous features.

Author(s):
    Quinn Gallagher, Ankur Gupta, Fanjin Wang, Christina Schenk

Created: 
    03/25/24
'''

class Surrogate:
    ''' 
        General surrogate class used for defining 
        new surrogate models.
    '''

    def __init__(self):
        self.name = 'General surrogate'
        self.model = None
        self.fitted = False

    def load_data(self, train_x, train_y):
        '''
            Stores new training data for fitting 
            this surrogate model.
        '''
        self.train_x = train_x
        self.train_y = train_y
        self.fitted = False

    def add_data(self, new_x, new_y):
        '''
            Adds new training data for fitting this
            surrogate model to the already loaded
            training data.
        '''
        self.train_x = np.vstack((
            self.train_x,
            new_x.reshape(-1, self.train_x.shape[1])
        ))
        self.train_y = np.hstack((
            self.train_y,
            new_y.reshape(-1)
        ))
        self.fitted = False

    def fit(self):
        '''
            Method for fitting self.model to the
            data that has been loaded into the surrogate
            class.
        '''
        raise NotImplementedError(
            f'\'Fit\' method not yet implemented for {self.__class__.__name__}'
        )
    
    def predict_means_and_stddevs(self, domain):
        '''
            Method for giving means and uncertainties
            to the provided domain using the model that
            has been fitted to the already loaded data.
        '''
        raise NotImplementedError(
            f'\'predict_means_and_stddevs\' method not yet implemented \
                for {self.__class__.__name__}'
        )

class RandomForestSurrogate(Surrogate):
    '''
        Surrogate class based on a Random Forest
        regressor from sklearn.
    '''

    def __init__(self):
        super().__init__()
        self.name = 'Random Forest surrogate'
        self.model = RandomForestRegressor()
        self.fitted = False

    def fit(self, ht=True, progress=False):

        self.fitted = True

        # Conduct hyperparameter tuning for optimal
        # model fitting.
        if ht:
            max_iter = 100
            cv_folds = 5
            patience = 10
            threshold = 0.05
            rf_reg = BayesSearchCV(
                estimator=RandomForestRegressor(),
                search_spaces={
                    'n_estimators': Integer(100, 300),
                    'max_features': Categorical(['sqrt', 'log2', None]),
                    'max_depth': Integer(10, 100),
                    'min_samples_split': Integer(2,6),
                    'min_samples_leaf': Integer(1,4) 
                },
                n_iter=max_iter,
                scoring='r2',
                n_jobs=1,
                n_points=1,
                cv=cv_folds,
                refit=True,
                verbose=0,
                random_state=1,
                error_score=0.0
            )
            rf_reg.fit(
                self.train_x, 
                self.train_y, 
                callback=HollowIterationsStopper(
                    n_iterations=patience, threshold=threshold
                )
            )
            self.model = rf_reg.best_estimator_
         
        # Just train the default instantiation of
        # the random forest regressor.
        else:
            self.model.fit(X=self.train_x, y=self.train_y)

    def predict_means_and_stddevs(self, domain):

        # If the model is not already fitted, ensure that
        # the model has been fitted to the provided train-
        # ing data.
        if not self.fitted:
            self.fit()

        # Get predictions from RF estimators.
        predictions = np.array([
            tree.predict(domain) for tree in self.model.estimators_
        ])
        means = np.mean(predictions, axis=0)
        stddevs = np.std(predictions, axis=0)
        return (means, stddevs)
    
class GPTanimotoSurrogate(Surrogate):
    '''
        GP-based surrogate model that operates on
        binary vectors.
    '''

    def __init__(self):
        super().__init__()
        self.name = 'GP Tanimoto surrogate'
        self.model = None # GP requires training data.
        self.fitted = False

    def load_data(self, train_x, train_y):
        '''
            Stores new training data for fitting 
            this surrogate model.
        '''
        super().load_data(train_x, train_y)

        # Instantiate model here, since GP requires
        # training data upon definition.
        self.model = TanimotoGP(
            train_x=torch.IntTensor(self.train_x),
            train_y=torch.FloatTensor(self.train_y)
        )

    def add_data(self, new_x, new_y):
        '''
            Adds new training data for fitting this
            surrogate model to the already loaded
            training data.
        '''
        super().add_data(new_x, new_y)

        # Instantiate a new model here, since GP 
        # requires training data upon definition.
        self.model = TanimotoGP(
            train_x=torch.IntTensor(self.train_x),
            train_y=torch.FloatTensor(self.train_y)
        )

    def fit(self, max_iter=1000, lr=0.1, cutoff=1e-4, progress=False):
        '''
            Tune hyperparameters for Gaussian process.
        '''

        self.fitted = True

        # Cast training data to tensors.
        X_train = torch.IntTensor(self.train_x)
        y_train = torch.FloatTensor(self.train_y)

        # Fit model hyperparameters.
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        loss_prev = 0.0
        for i in range(max_iter):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()

            # Check for early stopping.
            curr_loss = loss.item()
            if abs(curr_loss - loss_prev) < cutoff:
                break
            loss_prev = curr_loss

            # Report progress, if applicable.
            if progress:
                print('Iter %d/%d - Loss: %.3f, Noise: %.3f' % (
                    i + 1, max_iter, loss.item(),
                    self.model.likelihood.noise.item()
                ))

            optimizer.step()
            if hyper=True:
                for param_name, param in self.model.named_parameters():
                    print(f'Parameter name: {param_name:42} value = {param.item()}')

    def predict_means_and_stddevs(self, domain):

        # Fit model if not already fit.
        if not self.fitted:
            self.fit()
        
        # Cast domain to tensor.
        domain = torch.FloatTensor(domain)

        # Get likelihood from GP.
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(domain))
            means = predictions.mean.detach().numpy()
            stddevs = predictions.stddev.detach().numpy()
            return (means, stddevs)

    def likelihood(self, domain):

        # Fit model if not already fit.
        if not self.fitted:
            self.fit()

        # Cast domain to tensor.
        domain = torch.FloatTensor(domain)

        # Get likelihood from GP.
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            likelihood = self.model.likelihood(self.model(domain))
            return likelihood
        
class GPRQSurrogate(Surrogate):

    def __init__(self):
        super().__init__()
        self.name = 'GP RQ-Kernel Surrogate'
        self.model = None
        self.fitted = False
        self.scaler = None

    def load_data(self, train_x, train_y):
        '''
            Stores new training data for fitting 
            this surrogate model.
        '''
        super().load_data(train_x, train_y)

        # Fit scaler to available data.
        self.scaler = StandardScaler().fit(self.train_x)

        # Instantiate model here, since GP requires
        # training data upon definition.
        self.model = GPRQ(
            train_x=torch.FloatTensor(self.scaler.transform(self.train_x)),
            train_y=torch.FloatTensor(self.train_y)
        )

    def add_data(self, new_x, new_y):
        '''
            Adds new training data for fitting this
            surrogate model to the already loaded
            training data.
        '''
        super().add_data(new_x, new_y)

        # Fit scaler to newly added data.
        self.scaler = StandardScaler().fit(self.train_x)

        # Instantiate a new model here, since GP 
        # requires training data upon definition.
        self.model = GPRQ(
            train_x=torch.FloatTensor(self.scaler.transform(self.train_x)),
            train_y=torch.FloatTensor(self.train_y)
        )

    def fit(self, max_iter=1000, lr=0.1, cutoff=1e-4, progress=False):
        '''
            Tune hyperparameters for Gaussian process.
        '''

        self.fitted = True

        # Convert training data to suitable format.
        X_train = torch.FloatTensor(self.scaler.transform(self.train_x))
        y_train = torch.FloatTensor(self.train_y)

        # Fit model hyperparameters.
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        loss_prev = 0.0
        for i in range(max_iter):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()

            # Check for early stopping.
            curr_loss = loss.item()
            if abs(curr_loss - loss_prev) < cutoff:
                break
            loss_prev = curr_loss

            # Report progress, if applicable.
            if progress:
                print('Iter %d/%d - Loss: %.3f, Noise: %.3f' % (
                    i + 1, max_iter, loss.item(),
                    self.model.likelihood.noise.item()
                ))
            
            optimizer.step()

    def predict_means_and_stddevs(self, domain):

        # Fit model if not already fit.
        if not self.fitted:
            self.fit()
        
        # Cast domain to tensor.
        domain = torch.FloatTensor(self.scaler.transform(domain))

        # Get predictions from GP.
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(domain))
            means = predictions.mean.detach().numpy()
            stddevs = predictions.stddev.detach().numpy()
            return (means, stddevs)
    def likelihood(self, domain):

        # Fit model if not already fit.
        if not self.fitted:
            self.fit()

        # Cast domain to tensor.
        domain = torch.FloatTensor(domain)

        # Get predictions from GP.
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            likelihood = self.model.likelihood(self.model(domain))
            return likelihood
# =============================================
#   HELPER CLASSES / FUNCTIONS
# =============================================
        
# Helper classes / functions for Tanimoto GP from mol_opt GitHub repo.
# Source: https://github.com/wenhao-gao/mol_opt/blob/main/main/gpbo/gp/tanimoto_gp.py
        
def batch_tanimoto_sim(x1: torch.Tensor, x2: torch.Tensor):
    """tanimoto between two batched tensors, across last 2 dimensions"""
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2 ** 2, dim=-1, keepdims=True)
    return (dot_prod) / (x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod)

class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto coefficient kernel"""

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        return batch_tanimoto_sim(x1, x2)

class TanimotoGP(gpytorch.models.ExactGP):
    _num_outputs = 1

    def __init__(
        self,
        train_x,
        train_y,
        likelihood=None,
    ):

        # Fill in likelihood
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood

    def forward(self, x):

        # Normal mean + covar
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def hparam_dict(self):
        return {
            "likelihood.noise": self.likelihood.noise.item(),
            "covar_module.outputscale": self.covar_module.outputscale.item(),
            "mean_module.constant": self.mean_module.constant.item(),
        }

# Helper class for anisotropic RQ-kernel based GP. Implemented by Quinn using
# the ExactGP regression tutorial here:
# https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
class GPRQ(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPRQ, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(
            ard_num_dims=train_x.size(dim=1)
        ))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def acqf_EI(means, uncertainties, best_observed):
    '''
        Computes the Expected Improvement acquisition function
        for a given set of means and uncertainties.

    '''
    assert means.shape == uncertainties.shape
    z = torch.tensor((means - best_observed) / uncertainties, dtype=torch.float32)
    ei = torch.tensor(uncertainties, dtype=torch.float32) * (z * torch.distributions.Normal(0, 1).cdf(z) + torch.distributions.Normal(0, 1).log_prob(z).exp())
    return ei