import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def learn_truth_directions(acts_centered, labels, polarities):
    # Check if all polarities are zero (handling both int and float) -> if yes learn only t_g
    all_polarities_zero = torch.allclose(polarities, torch.tensor([0.0]), atol=1e-8)
    # Make the sure the labels only have the values -1.0 and 1.0
    labels_copy = labels.clone()
    labels_copy = torch.where(labels_copy == 0.0, torch.tensor(-1.0), labels_copy)
    
    if all_polarities_zero:
        X = labels_copy.reshape(-1, 1)
    else:
        X = torch.column_stack([labels_copy, labels_copy * polarities])

    # Compute the analytical OLS solution
    solution = torch.linalg.inv(X.T @ X) @ X.T @ acts_centered

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]

    return t_g, t_p

def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True, max_iter=10000)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc


class TTPD():
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True, max_iter=10000)
        probe.LR.fit(acts_2d, labels.numpy())
        return probe
    
    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return torch.tensor(self.LR.predict(acts_2d))
    
    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d
    
class LRProbe():

    def __init__(self):
        self.LR = None

    def from_data(acts, labels):
        probe = LRProbe()
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True,max_iter=10000)
        probe.LR.fit(acts.numpy(), labels.numpy())
        return probe

    def pred(self, acts):
        return torch.tensor(self.LR.predict(acts))
    
class LRProbePCA():
    """
    Logistic-regression probe that first gathers 100 first principal components from the training data, and then trains an LRC probe on that data. The test set is the projected on the same 100-dimensional subspace. 
    """

    def __init__(self):
        self.scaler: StandardScaler  
        self.pca: PCA  
        self.clf: LogisticRegression  
        self.LR: LogisticRegression  

    def from_data( acts: torch.Tensor, labels: torch.Tensor,
                  n_components: int = 100):

        probe = LRProbePCA()

        X = acts.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        probe.scaler = StandardScaler()
        X_scaled = probe.scaler.fit_transform(X)

        probe.pca = PCA(n_components=n_components, svd_solver="randomized")
        X_reduced = probe.pca.fit_transform(X_scaled)

        probe.LR = LogisticRegression(
            penalty=None,       # same as your original intent
            fit_intercept=True,
            max_iter=10000,
        )

        probe.LR.fit(X_reduced, y)

        return probe

 
    def pred(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for *new* activations (no refitting happens).
        """

        X = acts.detach().cpu().numpy()
        X_scaled = self.scaler.transform(X)      # <-- uses *frozen* params
        X_reduced = self.pca.transform(X_scaled)
        preds = self.LR.predict(X_reduced)

        return torch.tensor(preds, device=acts.device)