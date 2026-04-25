import numpy as np
import xgboost as xgb
import scipy.stats as stats
from typing import Tuple
from skfolio.distribution import GaussianCopula
import openturns as ot

def gauss_tau2par(tau):
    """
    Return Gaussian parameters from Kendall's tau values

    :param tau: Kendall's tau value

    :return: Gaussian copula parameter
    """
    return np.sin((np.pi/2)*tau)
    
def gauss_par2tau(par):
    """
    Return Kendall's tau from Gaussian copula parameter

    :param par: Gaussian copula parameter

    :return: Kendall's tau value
    """
    return (2/np.pi)*np.arcsin(par)

def get_initial_guess(y):
    """
    Return an initial guess for the prediction. This is loss-specific. We set the dependence parameters to Kendall's tau estimates for the entire dataset of pseudo-observations

    :param y: target matrix of the training set

    :return: np.ndarray with (transformed) parameter initial guess
    """
    #u0 = y[:, 0].reshape(-1,1)
    #u1 = y[:, 1].reshape(-1,1)
    #tau, _ = stats.kendalltau(u0, u1)

    gaussian_copula = GaussianCopula(itau=False)
    gaussian_copula.fit(y)
    
    return gaussian_link(gaussian_copula.rho_)

def sample_gaussian_copula(rho_array, seed):
    """
    Sample one bivariate observation from a Gaussian copula for each rho in rho_array.

    Parameters:
        rho_array (np.ndarray): shape (n,), correlation coefficients in (-1, 1)

    Returns:
        np.ndarray: shape (n, 2), samples from the Gaussian copula
    """
    ot.RandomGenerator.SetSeed(seed)
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

    samples = np.array(samples)
    # Clip to avoid 0 or 1
    eps = 1e-10
    samples = np.clip(samples, eps, 1 - eps)
    return samples
    
def gaussian_lpdf(u, rho):
    """
    Gaussian copula log-likelihood evaluation

    :param u: bivariate pseudo-observation vector (n_obs, 2)
    :param rho: current copula parameter estimates (n_obs, 1)
        
    :return: Gaussian copula log-likelihood
    """
    a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
    b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
    c = a*b
    d = a**2 + b**2
    return np.sum(-np.log(np.sqrt(1-rho**2)) - ((rho**2)*d - 2*rho*c)/(2*(1-rho**2)))

def gaussian_deviance(y, rho_pred):
    gaussian_cop = GaussianCopula(itau=False)
    rho_saturated = []
    for u in y:
        gaussian_cop.fit(u.reshape(-1,2))
        rho_saturated.append(gaussian_cop.rho_)
        
    rho_saturated = np.array(rho_saturated).reshape(-1,1)
    return 2*(gaussian_lpdf(y,rho_saturated)-gaussian_lpdf(y,rho_pred))

def gaussian_lpdf_grad(u, rho):
    """
    Gaussian copula gradient w.r.t copula parameter rho

    :param u: bivariate pseudo-observation vector (n_obs, 2)
    :param rho: current copula parameter estimates (n_obs, 1)
        
    :return: Gaussian copula lpdf gradient
    """
    a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
    b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
    c = a*b
    d = a**2 + b**2
    return rho/(1-rho**2) + ((1+rho**2)*c - d*rho)/((1-rho**2)**2)
    
def gaussian_link(par):
    """
    :param par: current parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    :return: link function of copula parameter
    """
    return np.tan((np.pi/2)*par)
    
def gaussian_inv_link(tpar):
    """
    :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    :return: inverse link function of transformed copula parameter
    """
    return (2/np.pi)*np.arctan(tpar)
    
def gaussian_inv_link_grad(tpar):
    """
    :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    return: gradient of inverse link function w.r.t transformed copula parameter
    """
    return (2/np.pi)*(1/(np.square(tpar)+1))

def gaussian_gradient(u: np.ndarray, predt: np.ndarray) -> np.ndarray:
    '''Compute the gradient of the nll'''
    #u = dtrain.get_label()
    #tau = gauss_par2tau(gaussian_inv_link(predt))
    #rho = gauss_tau2par(tau)
    rho = gaussian_inv_link(predt)
    return -gaussian_lpdf_grad(u, rho)*gaussian_inv_link_grad(predt)
        
def gaussian_copula_nll_obj(u: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Gaussian copula negative log-likelihood objective'''
    grad = gaussian_gradient(u, predt)
    hess = grad**2 #BHHH hessian substitute
    return grad, hess

def gaussian_copula_nll_metric(u: np.ndarray, predt: np.ndarray) -> Tuple[str, float]:
    ''' Gaussian copula negative log-likelihood metric'''
    #u = dtrain.get_label()
    #tau = gauss_par2tau(gaussian_inv_link(predt))
    #rho = gauss_tau2par(tau)
    rho = gaussian_inv_link(predt)
    log_likelihood = gaussian_lpdf(u, rho)
    return float(-log_likelihood)

def gauss_predt2tau(predt):
    return gauss_par2tau(gaussian_inv_link(predt))