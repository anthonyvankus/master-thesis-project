import numpy as np
import xgboost as xgb
import scipy.stats as stats
import scipy.special as special
import openturns as ot
from skfolio.distribution import ClaytonCopula
from typing import Tuple

def clayton_tau2par(tau):
    """
    Return Clayton parameters from Kendall's tau values
    """
    return 2*(tau/(1-tau))

def clayton_par2tau(par):
    """
    Return Kendall's tau from Clayton parameter values
    """
    return par/(par+2)

def get_initial_guess(y):
    """
    Return an initial guess for the prediction. This is loss-specific. We set the dependence parameters to Kendall's tau estimates for the entire dataset of pseudo-observations

    :param y: target matrix of the training set

    :return: np.ndarray with (transformed) parameter initial guess
    """
    #u0 = y[:, 0].reshape(-1,1)
    #u1 = y[:, 1].reshape(-1,1)
    #tau, _ = stats.kendalltau(u0, u1)

    clayton_copula = ClaytonCopula(itau=False)
    clayton_copula.fit(y)
    
    return special.logit(clayton_par2tau(clayton_copula.theta_))
    
    #return special.logit(tau)

def sample_clayton_copula(theta_array, seed):
    """
    Sample one point from a bivariate Clayton copula for each parameter value in theta_array.
    
    Parameters:
        theta_array (np.ndarray): 1D array of positive Clayton copula parameters (theta > 0).
    
    Returns:
        samples (np.ndarray): Array of shape (n, 2) where each row is a sample (u1, u2).
    """
    ot.RandomGenerator.SetSeed(seed)
    samples = []
    for theta in theta_array:
        copula = ot.ClaytonCopula(theta)
        sample = copula.getSample(1)
        samples.append(sample[0])

    samples = np.array(samples)
    eps = 1e-10
    samples = np.clip(samples, eps, 1 - eps)
    return samples

def clayton_Afunc(u, theta):
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)
    a = np.hstack((-theta*np.log(u), np.zeros_like(u0)))
    b = np.hstack((np.ones_like(u0), np.ones_like(u1), -np.ones_like(u0)))

    return np.exp(special.logsumexp(a, b=b, axis=1)).reshape(-1,1) #u0**(-theta) + u1**(-theta) - 1

def clayton_logAfunc(u, theta):
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)
    a = np.hstack((-theta*np.log(u), np.zeros_like(u0)))
    b = np.hstack((np.ones_like(u0), np.ones_like(u1), -np.ones_like(u0)))
    #print(a.shape)
    #print(b.shape)

    return special.logsumexp(a, b=b, axis=1).reshape(-1,1)
    
def clayton_Afunc_grad(u, theta):
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)
    return -((u0**(-theta))*np.log(u0) + (u1**(-theta))*np.log(u1))

def clayton_logA_grad(u, theta):
    A_func = clayton_Afunc(u, theta)
    return clayton_Afunc_grad(u, theta)/A_func

def clayton_lpdf(u, theta):
    """
    Clayton copula log-likelihood evaluation

    :param u: bivariate pseudo-observation vector (n_obs, 2)
    :param rho: current copula parameter estimates (n_obs, 1)
        
    :return: Clayton copula log-likelihood
    """
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)
    A_func = clayton_Afunc(u, theta)
    #print(np.log(A_func))
    #print(clayton_logAfunc(u, theta))
    lpdf = special.log1p(theta) - (1+theta)*(np.log(u0)+ np.log(u1)) - (2+1/theta)*clayton_logAfunc(u, theta) #np.log(A_func)
    return np.sum(lpdf)

def clayton_lpdf_grad(u, theta):
    """
    Clayton copula gradient w.r.t copula parameter theta

    :param u: bivariate pseudo-observation vector (n_obs, 2)
    :param rho: current copula parameter estimates (n_obs, 1)
        
    :return: Clayton copula lpdf gradient
    """
    u0 = u[:, 0].reshape(-1,1)
    u1 = u[:, 1].reshape(-1,1)

    #A_func = clayton_Afunc(u, theta)
    #logA = clayton_logAfunc(u, theta)
    dlogA = clayton_logA_grad(u, theta)
    lpdf_grad = 1/(1+theta) - (np.log(u0) + np.log(u1)) - (2 + 1/theta)*dlogA + clayton_logAfunc(u, theta)/np.square(theta)
    return lpdf_grad
    
def clayton_link(par):
    """
    :param par: current parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    :return: link function of copula parameter
    """
    return np.log(par)
    
def clayton_inv_link(tpar):
    """
    :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    :return: inverse link function of transformed copula parameter
    """
    return np.exp(tpar)
    
def clayton_inv_link_grad(tpar):
    """
    :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

    return: gradient of inverse link function w.r.t transformed copula parameter
    """
    return np.exp(tpar)

def clayton_gradient(u: np.ndarray, predt: np.ndarray) -> np.ndarray:
    '''Compute the gradient of the nll'''
    #u = dtrain.get_label()
    tau = special.expit(predt)
    #rho = gauss_tau2par(tau)
    theta = clayton_tau2par(tau)
    return -clayton_lpdf_grad(u, theta)*(2*tau/(1-tau)) # explain # *clayton_inv_link_grad(predt)
        
def clayton_copula_nll_obj(u: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Clayton copula negative log-likelihood objective'''
    grad = clayton_gradient(u, predt)
    hess = grad**2 #BHHH hessian substitute
    return grad, hess

def clayton_copula_nll_metric(u: np.ndarray, predt: np.ndarray) -> Tuple[str, float]:
    ''' Clayton copula negative log-likelihood metric'''
    #u = dtrain.get_label()
    tau = special.expit(predt)
    #rho = gauss_tau2par(tau)
    theta = clayton_tau2par(tau) #clayton_inv_link(predt)
    log_likelihood = clayton_lpdf(u, theta)
    return float(-log_likelihood)

def clayton_predt2tau(predt):
    return clayton_par2tau(special.expit(predt))