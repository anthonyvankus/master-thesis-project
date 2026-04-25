import numpy as np
import xgboost as xgb
import scipy.stats as stats
import scipy.special as special
import openturns as ot
from skfolio.distribution import GumbelCopula
from typing import Tuple

def gumbel_tau2par(tau):
    """
    Return Gumbel parameters from Kendall's tau values
    """
    return 1/(1-tau)

def gumbel_par2tau(par):
    """
    Return Kendall's tau from Gumbel parameter values
    """
    return 1-(1/par)

def get_initial_guess(y):
    """
    Return an initial guess for the prediction. This is loss-specific. We set the dependence parameters to Kendall's tau estimates for the entire dataset of pseudo-observations

    :param y: target matrix of the training set

    :return: np.ndarray with (transformed) parameter initial guess
    """
    #u0 = y[:, 0].reshape(-1,1)
    #u1 = y[:, 1].reshape(-1,1)
    #tau, _ = stats.kendalltau(u0, u1)

    gumbel_copula = GumbelCopula(itau=False)
    gumbel_copula.fit(y)
    
    return special.logit(gumbel_par2tau(gumbel_copula.theta_))

def sample_gumbel_copula(theta_array, seed):
    """
    Sample one observation from a bivariate Gumbel copula for each theta.
    Uses OpenTURNS for sampling.
    
    Parameters:
        theta_array (np.ndarray): array of Gumbel copula parameters (theta >= 1)
    
    Returns:
        np.ndarray: shape (n, 2), copula samples in (0, 1)
    """
    ot.RandomGenerator.SetSeed(seed)
    samples = []
    for theta in theta_array:
        copula = ot.GumbelCopula(theta)
        sample = copula.getSample(1)
        samples.append(sample[0])
    
    samples = np.array(samples)
    # Clip to avoid 0 or 1
    eps = 1e-10
    samples = np.clip(samples, eps, 1 - eps)
    return samples

def gumbel_lpdf(u, theta):
    """
    Gumbel copula log-likelihood evaluation

    :param u: bivariate pseudo-observation vector (n_obs, 2)
    :param rho: current copula parameter estimates (n_obs, 1)
        
    :return: Gumbel copula log-likelihood
    """
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)
    t = (-np.log(u0))**theta
    s = (-np.log(u1))**theta
    a = (t + s)**(1/theta)

    return np.sum(-a - np.log(u0*u1) + (-2+2/theta)*np.log(t+s) + (theta - 1)*(np.log(np.log(u0)*np.log(u1))) + np.log(1+(theta-1)/a))
    
def gumbel_lpdf_grad(u, theta):
    """
    Gumbel copula gradient w.r.t dependence parameter theta

    :param u: bivariate pseudo-observation vector (n_obs, 2)
    :param theta: current dependence parameter estimates (n_obs, 1)
        
    :return: Gumbel copula lpdf gradient
    """
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)
    t = (-np.log(u0))**theta
    s = (-np.log(u1))**theta
    a = (t + s)**(1/theta)
    b = np.log(t+s)/np.square(theta)
    d = (t*np.log(-np.log(u0)) + s*np.log(-np.log(u1)))/(t+s)

    return -a*(-b + d/theta) + (-2*b + (-2+2/theta)*d) + np.log(np.log(u0)*np.log(u1)) + a/(a + (theta - 1))*(1/a + ((theta-1)/a)*(b - d/theta))
    
def gumbel_link(par):
    """
    :param par: current parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    :return: link function of copula parameter
    """
    return np.log(par-1)
    
def gumbel_inv_link(tpar):
    """
    :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    :return: inverse link function of transformed copula parameter
    """
    return np.exp(tpar)+1
    
def gumbel_inv_link_grad(tpar):
    """
    :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    return: gradient of inverse link function w.r.t transformed copula parameter
    """
    return np.exp(tpar)

def gumbel_gradient(u: np.ndarray, predt: np.ndarray) -> np.ndarray:
    '''Compute the gradient of the nll'''
    #u = dtrain.get_label()
    #tau = gauss_par2tau(copula_inv_link(predt))
    #rho = gauss_tau2par(tau)
    tau = special.expit(predt)
    #rho = gauss_tau2par(tau)
    theta = gumbel_tau2par(tau)
    #theta = gumbel_inv_link(predt)
    return -gumbel_lpdf_grad(u, theta)*(2*tau/(1-tau)) #*gumbel_inv_link_grad(predt)
        
def gumbel_copula_nll_obj(u: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Gumbel copula negative log-likelihood objective'''
    grad = gumbel_gradient(u, predt)
    hess = grad**2 #BHHH hessian substitute
    return grad, hess

def gumbel_copula_nll_metric(u: np.ndarray, predt: np.ndarray) -> Tuple[str, float]:
    ''' Gumbel copula negative log-likelihood metric'''
    #u = dtrain.get_label()
    tau = special.expit(predt)
    #rho = gauss_tau2par(tau)
    theta = gumbel_tau2par(tau)
    #tau = gauss_par2tau(gaussian_inv_link(predt))
    #rho = gauss_tau2par(tau)
    #theta = gumbel_inv_link(predt)
    log_likelihood = gumbel_lpdf(u, theta)
    return float(-log_likelihood)

def gumbel_predt2tau(predt):
    return gumbel_par2tau(special.expit(predt))