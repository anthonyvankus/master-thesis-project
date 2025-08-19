import sys
import os

# Set project root relative to script location
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sklearn.linear_model import GammaRegressor
import mbtr.utils as ut
from scipy.linalg import hankel
from scipy.stats import expon
import scipy.special as special
import matplotlib.pyplot as plt
from mbtr.mbtr import MBT
from mbtr.utils import set_figure
from mbtr import LOSS_MAP
import Losses
from Losses import BiGaussCopLoss
from Losses.BiGaussCopLoss import BiGaussCopLikelihood
from Losses import BiClaytonCopLoss
from Losses.BiClaytonCopLoss import BiClaytonCopLikelihood
from Losses import BiGumbelCopLoss
from Losses.BiGumbelCopLoss import BiGumbelCopLikelihood
from Losses import ExpLoss
from Losses.ExpLoss import JointExpLikelihood
import scipy.stats as stats
import openturns as ot
import json
from contextlib import contextmanager

# for the MBT class to recognize the subclass as a type of loss
LOSS_MAP['gaussian'] = BiGaussCopLikelihood
LOSS_MAP['clayton'] = BiClaytonCopLikelihood
LOSS_MAP['gumbel'] = BiGumbelCopLikelihood
LOSS_MAP['joint_exponential'] = JointExpLikelihood

jointexp = JointExpLikelihood()
clayton = BiClaytonCopLikelihood()
gumbel = BiGumbelCopLikelihood()
gaussian = BiGaussCopLikelihood()

def clayton_tau2par(tau):
        """
        Return Clayton parameters from Kendall's tau values
        """
        return 2*(tau/(1-tau))

def gaussian_tau2par(tau):
        """
        Return Gaussian parameters from Kendall's tau values
        """
        return np.sin((np.pi/2)*tau)

def gumbel_tau2par(tau):
        """
        Return Gumbel parameters from Kendall's tau values
        """
        return 1/(1-tau)

def sample_clayton_copula(theta_array):
    """
    Sample one point from a bivariate Clayton copula for each parameter value in theta_array.
    
    Parameters:
        theta_array (np.ndarray): 1D array of positive Clayton copula parameters (theta > 0).
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        samples (np.ndarray): Array of shape (n, 2) where each row is a sample (u1, u2).
    """
    samples = []
    for theta in theta_array:
        copula = ot.ClaytonCopula(theta)
        sample = copula.getSample(1)
        samples.append(sample[0])

    samples = np.array(samples)
    eps = 1e-10
    samples = np.clip(samples, eps, 1 - eps)
    return samples

def sample_gaussian_copula(rho_array):
    """
    Sample one bivariate observation from a Gaussian copula for each rho in rho_array.

    Parameters:
        rho_array (np.ndarray): shape (n,), correlation coefficients in (-1, 1)

    Returns:
        np.ndarray: shape (n, 2), samples from the Gaussian copula
    """
    n = len(rho_array)
    samples = np.zeros((n, 2))

    for i, rho in enumerate(rho_array):
        if not (-1 < rho < 1):
            raise ValueError("Each rho must be in the interval (-1, 1).")

        # Define correlation matrix
        corr_matrix = ot.CorrelationMatrix(2)
        corr_matrix[0, 1] = rho

        # Create the Gaussian copula
        copula = ot.NormalCopula(corr_matrix)

        # Draw a single sample
        u = copula.getSample(1)[0]
        samples[i] = [u[0], u[1]]

    return samples

def sample_gumbel_copula(theta_array):
    """
    Sample one observation from a bivariate Gumbel copula for each theta.
    Uses OpenTURNS for sampling.
    
    Parameters:
        theta_array (np.ndarray): array of Gumbel copula parameters (theta >= 1)
    
    Returns:
        np.ndarray: shape (n, 2), copula samples in (0, 1)
    """
    samples = []
    for theta in theta_array:
        copula = ot.GumbelCopula(theta)
        sample = copula.getSample(1)
        samples.append(sample[0])
    
    samples = np.array(samples)
    # Clip to avoid 0 or 1 (required for exponential transform)
    eps = 1e-10
    samples = np.clip(samples, eps, 1 - eps)
    return samples

def copula_to_exponential(copula_samples, lambda1, lambda2):
    """
    Transform copula samples to exponential margins with variable rate parameters.
    
    Parameters:
        copula_samples (np.ndarray): shape (n, 2), values in (0, 1)
        lambda1 (np.ndarray): shape (n,), rate parameters for X1
        lambda2 (np.ndarray): shape (n,), rate parameters for X2
    
    Returns:
        np.ndarray: shape (n, 2), dependent exponential samples
    """
    u1, u2 = copula_samples[:, 0], copula_samples[:, 1]
    lambda1 = np.asarray(lambda1)
    lambda2 = np.asarray(lambda2)

    if np.any(u1 <= 0) or np.any(u2 <= 0):
        raise ValueError("Copula samples must be in (0, 1), not inclusive.")
    if np.any(lambda1 <= 0) or np.any(lambda2 <= 0):
        raise ValueError("Rate parameters must be > 0.")

    y1 = expon.ppf(u1, scale = 1/lambda1)
    y2 = expon.ppf(u2, scale = 1/lambda2) 

    return np.column_stack((y1, y2))

def clip_copula_samples(samples, eps=1e-10):
    """
    Ensure copula samples are strictly inside (0, 1)^2.
    
    Parameters:
        samples (np.ndarray): shape (n, 2), values in [0, 1]
        eps (float): small buffer to keep samples strictly in (0, 1)
    
    Returns:
        np.ndarray: clipped samples in (eps, 1 - eps)
    """
    return np.clip(samples, eps, 1 - eps)

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# Step-wise function
n_sim = 500

benchmark_marginal_results_steep = {
    "mle": {"clayton_sample": {"mse": [], "-logL": []}, "gumbel_sample": {"mse": [], "-logL": []}, "gaussian_sample": {"mse": [], "-logL": []}},
    "glm": {"clayton_sample": {"mse": [], "-logL": []}, "gumbel_sample": {"mse": [], "-logL": []}, "gaussian_sample": {"mse": [], "-logL": []}}
}
#benchmark_marginal_results_steep = {"clayton_sample": {"mse": [], "-logL": []}, "gumbel_sample": {"mse": [], "-logL": []}, "gaussian_sample": {"mse": [], "-logL": []}}
benchmark_copula_results_steep = {"true": {"clayton": {"mse": [], "-logL": []}, "gumbel": {"mse": [], "-logL": []}, "gaussian": {"mse": [], "-logL": []}}, "pseudo": {"clayton": {"mse": [], "-logL": []}, "gumbel": {"mse": [], "-logL": []}, "gaussian": {"mse": [], "-logL": []}}}
model_marginal_results_steep = {"clayton": {"mse": [], "-logL": []}, "gumbel": {"mse": [], "-logL": []}, "gaussian": {"mse": [], "-logL": []}}
model_copula_results_steep = {"true": {"clayton": {"mse": [], "-logL": []}, "gumbel": {"mse": [], "-logL": []}, "gaussian": {"mse": [], "-logL": []}}, "pseudo": {"clayton": {"mse": [], "-logL": []}, "gumbel": {"mse": [], "-logL": []}, "gaussian": {"mse": [], "-logL": []}}}

n = 10000
np.random.seed(42)

for i in range(n_sim):
    x = np.random.uniform(0, 1, (n, 3))
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    tau = 0.5 - 0.2/(1+np.exp(-40*(x1-0.4))) + 0.4/(1+np.exp(-40*(x2-0.75))) # steep sigmoid function

    clayton_sample = clip_copula_samples(sample_clayton_copula(clayton_tau2par(tau)))
    gaussian_sample = clip_copula_samples(sample_gaussian_copula(gaussian_tau2par(tau)))
    gumbel_sample = clip_copula_samples(sample_gumbel_copula(gumbel_tau2par(tau)))

    rate1 = np.exp(-1+3*x1 - 0.5*x3 + x1*(x3 > 0.6))
    rate2 = np.exp(-1-0.1*x1 + 2.5*x3 - x3*(x1 < 0.4))
    rate = np.stack((rate1, rate2), axis = 1)

    y_clayton = copula_to_exponential(clayton_sample, rate1, rate2)
    y_gumbel = copula_to_exponential(gumbel_sample, rate1, rate2)
    y_gauss = copula_to_exponential(gaussian_sample, rate1, rate2)

    # GLM marginal model
    glm_clayton = GammaRegressor(alpha=0.0, max_iter = 500)
    glm_gumbel = GammaRegressor(alpha=0.0, max_iter = 500)
    glm_gaussian = GammaRegressor(alpha=0.0, max_iter = 500)
    
    interaction_effect = (x1*x3).reshape(-1,1)
    X = np.hstack((x, interaction_effect))
    
    glm_clayton.fit(X,  y_clayton[:, 0]) 
    glm_gumbel.fit(X, y_gumbel[:, 0])
    glm_gaussian.fit(X, y_gauss[:, 0])
    
    glm_clayton_pred1 = 1/glm_clayton.predict(X)
    glm_gumbel_pred1 = 1/glm_gumbel.predict(X)
    glm_gaussian_pred1 = 1/glm_gaussian.predict(X)
    
    glm_clayton.fit(X, y_clayton[:, 1])
    glm_gumbel.fit(X, y_gumbel[:, 1])
    glm_gaussian.fit(X, y_gauss[:, 1])
    
    glm_clayton_pred2 = 1/glm_clayton.predict(X)
    glm_gumbel_pred2 = 1/glm_gumbel.predict(X)
    glm_gaussian_pred2 = 1/glm_gaussian.predict(X)
    
    glm_clayton_pred = np.hstack((glm_clayton_pred1.reshape(-1,1), glm_clayton_pred2.reshape(-1,1)))
    glm_gumbel_pred = np.hstack((glm_gumbel_pred1.reshape(-1,1), glm_gumbel_pred2.reshape(-1,1)))
    glm_gaussian_pred = np.hstack((glm_gaussian_pred1.reshape(-1,1), glm_gaussian_pred2.reshape(-1,1)))

    with suppress_stdout_stderr(): # suppress progress bar
        marginal_clayton = MBT(loss_type = "joint_exponential", n_boosts = 500, min_leaf = 2000, refit=False, learning_rate=0.1, lambda_weights=0.1).fit(x, y_clayton, do_plot=False)
        marginal_gumbel = MBT(loss_type = "joint_exponential", n_boosts = 500, min_leaf = 2000, refit=False, learning_rate=0.1, lambda_weights=0.1).fit(x, y_gumbel, do_plot=False)
        marginal_gaussian = MBT(loss_type = "joint_exponential", n_boosts = 500, min_leaf = 2000, refit=False, learning_rate=0.1, lambda_weights=0.1).fit(x, y_gauss, do_plot=False)

    nat_rate_clayton = marginal_clayton.predict(x)
    nat_rate_gumbel = marginal_gumbel.predict(x)
    nat_rate_gaussian = marginal_gaussian.predict(x)

    rate_clayton = np.hstack((np.exp(nat_rate_clayton[:,0].reshape(-1, 1)), np.exp(nat_rate_clayton[:,1].reshape(-1,1))))
    rate_gumbel = np.hstack((np.exp(nat_rate_gumbel[:,0].reshape(-1, 1)), np.exp(nat_rate_gumbel[:,1].reshape(-1,1))))
    rate_gaussian = np.hstack((np.exp(nat_rate_gaussian[:,0].reshape(-1, 1)), np.exp(nat_rate_gaussian[:,1].reshape(-1,1))))

    rate_bar_clayton = np.exp(jointexp.get_initial_guess(y_clayton))
    rate_bar_gumbel = np.exp(jointexp.get_initial_guess(y_gumbel))
    rate_bar_gauss = np.exp(jointexp.get_initial_guess(y_gauss))

    # Benchmark MLE marginal MSE
    benchmark_marginal_results_steep["mle"]["clayton_sample"]["mse"].append(mse(rate, rate_bar_clayton))
    benchmark_marginal_results_steep["mle"]["gumbel_sample"]["mse"].append(mse(rate, rate_bar_gumbel))
    benchmark_marginal_results_steep["mle"]["gaussian_sample"]["mse"].append(mse(rate, rate_bar_gauss))

    # Benchmark GLM marginal MSE
    benchmark_marginal_results_steep["glm"]["clayton_sample"]["mse"].append(mse(rate, glm_clayton_pred))
    benchmark_marginal_results_steep["glm"]["gumbel_sample"]["mse"].append(mse(rate, glm_gumbel_pred))
    benchmark_marginal_results_steep["glm"]["gaussian_sample"]["mse"].append(mse(rate, glm_gaussian_pred))

    # Benchmark MLE marginal -logL
    benchmark_marginal_results_steep["mle"]["clayton_sample"]["-logL"].append(jointexp.tree_loss(y_clayton, np.log(rate_bar_clayton)))
    benchmark_marginal_results_steep["mle"]["gumbel_sample"]["-logL"].append(jointexp.tree_loss(y_gumbel, np.log(rate_bar_gumbel)))
    benchmark_marginal_results_steep["mle"]["gaussian_sample"]["-logL"].append(jointexp.tree_loss(y_gauss, np.log(rate_bar_gauss)))

    # Benchmark GLM marginal -logL
    benchmark_marginal_results_steep["glm"]["clayton_sample"]["-logL"].append(jointexp.tree_loss(y_clayton, np.log(glm_clayton_pred)))
    benchmark_marginal_results_steep["glm"]["gumbel_sample"]["-logL"].append(jointexp.tree_loss(y_gumbel, np.log(glm_gumbel_pred)))
    benchmark_marginal_results_steep["glm"]["gaussian_sample"]["-logL"].append(jointexp.tree_loss(y_gauss, np.log(glm_gaussian_pred)))

    # Model marginal MSE
    model_marginal_results_steep["clayton"]["mse"].append(mse(rate, rate_clayton))
    model_marginal_results_steep["gumbel"]["mse"].append(mse(rate, rate_gumbel))
    model_marginal_results_steep["gaussian"]["mse"].append(mse(rate, rate_gaussian))

    # Model marginal -logL
    model_marginal_results_steep["clayton"]["-logL"].append(jointexp.tree_loss(y_clayton, np.log(rate_clayton)))
    model_marginal_results_steep["gumbel"]["-logL"].append(jointexp.tree_loss(y_gumbel, np.log(rate_gumbel)))
    model_marginal_results_steep["gaussian"]["-logL"].append(jointexp.tree_loss(y_gauss, np.log(rate_gaussian)))

    with suppress_stdout_stderr():
        # fit copula to true copula observations (U)
        mcop_clayton = MBT(n_boosts = 500, learning_rate = 0.1, loss_type = "clayton", refit = False, min_leaf = 2000, verbose = 0, lambda_weights = 0.1).fit(x, clayton_sample, do_plot = False)
        mcop_gumbel = MBT(n_boosts = 500, learning_rate = 0.1, loss_type = "gumbel", refit = False, min_leaf = 2000, verbose = 0, lambda_weights=0.1).fit(x, gumbel_sample, do_plot = False)
        mcop_gaussian = MBT(n_boosts = 500, learning_rate = 0.1, loss_type = "gaussian", refit = False, min_leaf = 2000, verbose = 0, lambda_weights=0.1).fit(x, gaussian_sample, do_plot = False)

    reparam_theta_clayton = mcop_clayton.predict(x)
    reparam_theta_gumbel = mcop_gumbel.predict(x)
    reparam_theta_gaussian = mcop_gaussian.predict(x)

    tau_clayton = special.expit(reparam_theta_clayton)
    tau_gumbel = special.expit(reparam_theta_gumbel)
    tau_gaussian = special.expit(reparam_theta_gaussian)

    tau_bar_clayton = special.expit(clayton.get_initial_guess(clayton_sample))
    tau_bar_gumbel = special.expit(gumbel.get_initial_guess(gumbel_sample))
    tau_bar_gaussian = special.expit(gaussian.get_initial_guess(gaussian_sample))

    # Benchmark copula MSE (U)
    benchmark_copula_results_steep["true"]["clayton"]["mse"].append(mse(tau.reshape(-1, 1), tau_bar_clayton))
    benchmark_copula_results_steep["true"]["gumbel"]["mse"].append(mse(tau.reshape(-1, 1), tau_bar_gumbel))
    benchmark_copula_results_steep["true"]["gaussian"]["mse"].append(mse(tau.reshape(-1, 1), tau_bar_gaussian))

    # Benchmark copula -logL (U)
    benchmark_copula_results_steep["true"]["clayton"]["-logL"].append(clayton.tree_loss(clayton_sample, special.logit(tau_bar_clayton)))
    benchmark_copula_results_steep["true"]["gumbel"]["-logL"].append(gumbel.tree_loss(gumbel_sample, special.logit(tau_bar_gumbel)))
    benchmark_copula_results_steep["true"]["gaussian"]["-logL"].append(gaussian.tree_loss(gaussian_sample, special.logit(tau_bar_gaussian)))

    # Model copula MSE (U)
    model_copula_results_steep["true"]["clayton"]["mse"].append(mse(tau.reshape(-1, 1), tau_clayton))
    model_copula_results_steep["true"]["gumbel"]["mse"].append(mse(tau.reshape(-1, 1), tau_gumbel))
    model_copula_results_steep["true"]["gaussian"]["mse"].append(mse(tau.reshape(-1, 1), tau_gaussian))

    # Model copula -logL (U)
    model_copula_results_steep["true"]["clayton"]["-logL"].append(clayton.tree_loss(clayton_sample, special.logit(tau_clayton)))
    model_copula_results_steep["true"]["gumbel"]["-logL"].append(gumbel.tree_loss(gumbel_sample, special.logit(tau_gumbel)))
    model_copula_results_steep["true"]["gaussian"]["-logL"].append(gaussian.tree_loss(gaussian_sample, special.logit(tau_gaussian)))

    # compute pseudo-observations based on estimated marginal CDF
    cdf_obs_clayton1 = expon.cdf(y_clayton[:, 0], scale = 1/rate_clayton[:, 0]).reshape(-1,1)
    cdf_obs_clayton2 = expon.cdf(y_clayton[:, 1], scale = 1/rate_clayton[:, 1]).reshape(-1,1)
    cdf_obs_clayton = np.hstack((cdf_obs_clayton1, cdf_obs_clayton2))

    cdf_obs_gumbel1 = expon.cdf(y_gumbel[:, 0], scale = 1/rate_gumbel[:, 0]).reshape(-1,1)
    cdf_obs_gumbel2 = expon.cdf(y_gumbel[:, 1], scale = 1/rate_gumbel[:, 1]).reshape(-1,1)
    cdf_obs_gumbel = np.hstack((cdf_obs_gumbel1, cdf_obs_gumbel2))

    cdf_obs_gaussian1 = expon.cdf(y_gauss[:, 0], scale = 1/rate_gaussian[:, 0]).reshape(-1,1)
    cdf_obs_gaussian2 = expon.cdf(y_gauss[:, 1], scale = 1/rate_gaussian[:, 1]).reshape(-1,1)
    cdf_obs_gaussian = np.hstack((cdf_obs_gaussian1, cdf_obs_gaussian2))

    with suppress_stdout_stderr():
         # fit copula to cdf-based pseudo copula observations (V)
        cdf_cop_clayton = MBT(n_boosts = 500, learning_rate = 0.1, loss_type = "clayton", refit = False, min_leaf = 2000, lambda_weights=0.1).fit(x, cdf_obs_clayton, do_plot = False)
        cdf_cop_gumbel = MBT(n_boosts = 500, learning_rate = 0.1, loss_type = "gumbel", refit = False, min_leaf = 2000, lambda_weights=0.1).fit(x, cdf_obs_gumbel, do_plot = False)
        cdf_cop_gaussian = MBT(n_boosts = 500, learning_rate = 0.1, loss_type = "gaussian", refit = False, min_leaf = 2000, lambda_weights=0.1).fit(x, cdf_obs_gaussian, do_plot = False)
    
    cdf_reparam_theta_clayton = cdf_cop_clayton.predict(x)
    cdf_reparam_theta_gumbel = cdf_cop_gumbel.predict(x)
    cdf_reparam_theta_gaussian = cdf_cop_gaussian.predict(x)

    cdf_tau_clayton = special.expit(cdf_reparam_theta_clayton)
    cdf_tau_gumbel = special.expit(cdf_reparam_theta_gumbel)
    cdf_tau_gaussian = special.expit(cdf_reparam_theta_gaussian)

    cdf_tau_bar_clayton = special.expit(clayton.get_initial_guess(cdf_obs_clayton))
    cdf_tau_bar_gumbel = special.expit(gumbel.get_initial_guess(cdf_obs_gumbel))
    cdf_tau_bar_gaussian = special.expit(gaussian.get_initial_guess(cdf_obs_gaussian))

    # Benchmark copula MSE (V)
    benchmark_copula_results_steep["pseudo"]["clayton"]["mse"].append(mse(tau.reshape(-1, 1), cdf_tau_bar_clayton))
    benchmark_copula_results_steep["pseudo"]["gumbel"]["mse"].append(mse(tau.reshape(-1, 1), cdf_tau_bar_gumbel))
    benchmark_copula_results_steep["pseudo"]["gaussian"]["mse"].append(mse(tau.reshape(-1, 1), cdf_tau_bar_gaussian))

    # Benchmark copula -logL (V)
    benchmark_copula_results_steep["pseudo"]["clayton"]["-logL"].append(clayton.tree_loss(clayton_sample, special.logit(cdf_tau_bar_clayton)))
    benchmark_copula_results_steep["pseudo"]["gumbel"]["-logL"].append(gumbel.tree_loss(gumbel_sample, special.logit(cdf_tau_bar_gumbel)))
    benchmark_copula_results_steep["pseudo"]["gaussian"]["-logL"].append(gaussian.tree_loss(gaussian_sample, special.logit(cdf_tau_bar_gaussian)))

    # Model copula MSE (V)
    model_copula_results_steep["pseudo"]["clayton"]["mse"].append(mse(tau.reshape(-1, 1), cdf_tau_clayton))
    model_copula_results_steep["pseudo"]["gumbel"]["mse"].append(mse(tau.reshape(-1, 1), cdf_tau_gumbel))
    model_copula_results_steep["pseudo"]["gaussian"]["mse"].append(mse(tau.reshape(-1, 1), cdf_tau_gaussian))

    # Model copula -logL (V)
    model_copula_results_steep["pseudo"]["clayton"]["-logL"].append(clayton.tree_loss(clayton_sample, special.logit(cdf_tau_clayton)))
    model_copula_results_steep["pseudo"]["gumbel"]["-logL"].append(gumbel.tree_loss(gumbel_sample, special.logit(cdf_tau_gumbel)))
    model_copula_results_steep["pseudo"]["gaussian"]["-logL"].append(gaussian.tree_loss(gaussian_sample, special.logit(cdf_tau_gaussian)))

# write to json file
with open('bench_marginal_results_steep.json', 'w', encoding='utf-8') as f:
    json.dump(benchmark_marginal_results_steep, f, ensure_ascii=False, indent=4)

with open('model_marginal_results_steep.json', 'w', encoding='utf-8') as f:
    json.dump(model_marginal_results_steep, f, ensure_ascii=False, indent=4)

with open('bench_copula_results_steep.json', 'w', encoding='utf-8') as f:
    json.dump(benchmark_copula_results_steep, f, ensure_ascii=False, indent=4)

with open('model_copula_results_steep.json', 'w', encoding='utf-8') as f:
    json.dump(model_copula_results_steep, f, ensure_ascii=False, indent=4)
