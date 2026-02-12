import sys
import os

# Set project root relative to script location
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import scipy
import statsmodels.api as sm
import mbtr.utils as ut
import scipy.special as special
import matplotlib.pyplot as plt
from mbtr.mbtr import MBT
from mbtr.utils import set_figure
from mbtr import LOSS_MAP
from Losses import GammaLoss
from Losses.GammaLoss import GammaLikelihood
from Losses import LognormalLoss
from Losses.LognormalLoss import LognormalLikelihood
import json
from contextlib import contextmanager

LOSS_MAP['gamma'] = GammaLikelihood
LOSS_MAP['lognormal'] = LognormalLikelihood

gamma_loss = GammaLikelihood()
lognormal_loss = LognormalLikelihood()

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

def bias(y, y_pred):
    return np.mean(y-y_pred)

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

results_spec0 = {
    "mle": {"gamma": {"alpha": {"mse": [], "mae": [], "bias": []}, "theta": {"mse": [], "mae": [], "bias": []}, "nll": []}, "lognormal": {"mu": {"mse": [], "mae": [], "bias": []}, "sigma": {"mse": [], "mae": [], "bias": []}, "nll": []}},
    "glm": {"gamma": {"alpha": {"mse": [], "mae": [], "bias": []}, "theta": {"mse": [], "mae": [], "bias": []}, "nll": []}, "lognormal": {"mu": {"mse": [], "mae": [], "bias": []}, "sigma": {"mse": [], "mae": [], "bias": []}, "nll": []}},
    "mbt": {"gamma": {"alpha": {"mse": [], "mae": [], "bias": []}, "theta": {"mse": [], "mae": [], "bias": []}, "nll": []}, "lognormal": {"mu": {"mse": [], "mae": [], "bias": []}, "sigma": {"mse": [], "mae": [], "bias": []}, "nll": []}}
}

results_spec2 = {
    "mle": {"gamma": {"alpha": {"mse": [], "mae": [], "bias": []}, "theta": {"mse": [], "mae": [], "bias": []}, "nll": []}, "lognormal": {"mu": {"mse": [], "mae": [], "bias": []}, "sigma": {"mse": [], "mae": [], "bias": []}, "nll": []}},
    "glm": {"gamma": {"alpha": {"mse": [], "mae": [], "bias": []}, "theta": {"mse": [], "mae": [], "bias": []}, "nll": []}, "lognormal": {"mu": {"mse": [], "mae": [], "bias": []}, "sigma": {"mse": [], "mae": [], "bias": []}, "nll": []}},
    "mbt": {"gamma": {"alpha": {"mse": [], "mae": [], "bias": []}, "theta": {"mse": [], "mae": [], "bias": []}, "nll": []}, "lognormal": {"mu": {"mse": [], "mae": [], "bias": []}, "sigma": {"mse": [], "mae": [], "bias": []}, "nll": []}}
}

n_sim = 500
n = 10000

for i in range(n_sim):
    np.random.seed(i)

    x = np.random.uniform(0, 1, (n, 5)) # x4 and x5 are included as feature noise
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    # simple linear relationship (scenario 1)
    # gamma:
    alpha0 = np.exp(0.5 + x1 + 1.5*x2 + 2.0*x3)
    theta0 = np.exp(1.0 + 0.75*x1 + 0.5*x2 + 0.25*x3)

    #lognormal:
    mu0 = 2 + x1 + 1.5*x2 + 2.0*x3
    sigma0 = np.exp(0.25 + 0.2*x1 + 0.1*x2 + 0.01*x3)

    # highly non-linear relationship with strong interaction effects (scenario 2)
    # gamma:
    alpha2 = np.exp(0.5 + 1.0*(x1 < 0.3) + 1.5*np.sin(2*np.pi*x2) + 2.0*x3**2 + 2.5*x3*(x2 > 0.5))
    theta2 = np.exp(0.2 + 0.15*np.exp(-x1**2) + 0.1*np.log(1 + np.exp(3*x2)) + 0.05*1/(1 + np.exp(-x1 + x3)))

    # lognormal:
    mu2 = 0.5 + 1.0*(x1 < 0.3) + 1.5*np.sin(2*np.pi*x2) + 2.0*x3**2 + 2.5*x3*(x2 > 0.5)
    sigma2 = np.exp(0.2 + 0.15*np.exp(-x1**2) + 0.1*np.log(1 + np.exp(3*x2)) + 0.05*1/(1 + np.exp(-x1 + x3)))

    # simulate from scenario 1
    gamma0 = scipy.stats.gamma.rvs(a = alpha0, scale = theta0, loc = 0, size=(1,n)).reshape(-1,1)
    lognormal0 = scipy.stats.lognorm.rvs(s = sigma0, scale = np.exp(mu0), loc = 0, size=(1,n)).reshape(-1,1)

    # fit MBT and GLM models
    with suppress_stdout_stderr(): # suppress progress bar
        # gamma and lognormal MBT fit
        gamma0_mbt = MBT(n_boosts = 2000, learning_rate = 0.2, loss_type = "gamma", refit = False, min_leaf = 2500, lambda_weights=0.1).fit(x, gamma0, do_plot = False)
        lognormal0_mbt = MBT(n_boosts = 2000, learning_rate = 0.2, loss_type = "lognormal", refit = False, min_leaf = 2500, lambda_weights=0.1).fit(x, lognormal0, do_plot = False)
        # gamma and lognormal GLM fit
        gamma0_glm = sm.GLM(gamma0, x, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        x_intercept = sm.add_constant(x)
        lognormal0_glm = sm.OLS(np.log(lognormal0), x_intercept).fit()

    # MBT predictions
    gamma0_mbt_pred = gamma0_mbt.predict(x)
    lognormal0_mbt_pred = lognormal0_mbt.predict(x)

    alpha0_mbt = np.exp(gamma0_mbt_pred[:,0]).reshape(-1,1)
    theta0_mbt = 1/np.exp(gamma0_mbt_pred[:,1]).reshape(-1,1)

    sigma0_mbt = np.sqrt(1/(2*np.exp(lognormal0_mbt_pred[:,1])).reshape(-1,1))
    mu0_mbt = lognormal0_mbt_pred[:,0].reshape(-1,1)*sigma0_mbt

    # GLM predictions
    alpha0_glm = 1/gamma0_glm.scale # scalar
    theta0_glm = (gamma0_glm.fittedvalues*gamma0_glm.scale).reshape(-1,1) # vector

    mu0_glm = lognormal0_glm.fittedvalues.reshape(-1,1) # vector
    sigma0_glm = np.sqrt(lognormal0_glm.scale) # scalar

    # fit MLE (intercept-only) model
    alpha0_mle, _, theta0_mle = scipy.stats.gamma.fit(gamma0, floc=0)
    sigma0_mle, _, mu0_mle = scipy.stats.lognorm.fit(lognormal0, floc = 0)
    mu0_mle = np.log(mu0_mle)

    # performance measures
    # MLE
    results_spec0["mle"]["gamma"]["alpha"]["mse"].append(mse(alpha0, alpha0_mle))
    results_spec0["mle"]["gamma"]["theta"]["mse"].append(mse(theta0, theta0_mle))
    results_spec0["mle"]["gamma"]["alpha"]["mae"].append(mae(alpha0, alpha0_mle))
    results_spec0["mle"]["gamma"]["theta"]["mae"].append(mae(theta0, theta0_mle))
    results_spec0["mle"]["gamma"]["alpha"]["bias"].append(bias(alpha0, alpha0_mle))
    results_spec0["mle"]["gamma"]["theta"]["bias"].append(bias(theta0, theta0_mle))
    
    results_spec0["mle"]["lognormal"]["mu"]["mse"].append(mse(mu0, mu0_mle))
    results_spec0["mle"]["lognormal"]["sigma"]["mse"].append(mse(sigma0, sigma0_mle))
    results_spec0["mle"]["lognormal"]["mu"]["mae"].append(mae(mu0, mu0_mle))
    results_spec0["mle"]["lognormal"]["sigma"]["mae"].append(mae(sigma0, sigma0_mle))
    results_spec0["mle"]["lognormal"]["mu"]["bias"].append(bias(mu0, mu0_mle))
    results_spec0["mle"]["lognormal"]["sigma"]["bias"].append(bias(sigma0, sigma0_mle))

    results_spec0["mle"]["gamma"]["nll"].append(-np.sum(scipy.stats.gamma.logpdf(x = gamma0, a = alpha0_mle, scale = theta0_mle)))
    results_spec0["mle"]["lognormal"]["nll"].append(-np.sum(scipy.stats.lognorm.logpdf(x = lognormal0, scale = np.exp(mu0_mle), s = sigma0_mle)))

    #GLM
    results_spec0["glm"]["gamma"]["alpha"]["mse"].append(mse(alpha0, alpha0_glm))
    results_spec0["glm"]["gamma"]["theta"]["mse"].append(mse(theta0, theta0_glm))
    results_spec0["glm"]["gamma"]["alpha"]["mae"].append(mae(alpha0, alpha0_glm))
    results_spec0["glm"]["gamma"]["theta"]["mae"].append(mae(theta0, theta0_glm))
    results_spec0["glm"]["gamma"]["alpha"]["bias"].append(bias(alpha0, alpha0_glm))
    results_spec0["glm"]["gamma"]["theta"]["bias"].append(bias(theta0, theta0_glm))
    
    results_spec0["glm"]["lognormal"]["mu"]["mse"].append(mse(mu0, mu0_glm))
    results_spec0["glm"]["lognormal"]["sigma"]["mse"].append(mse(sigma0, sigma0_glm))
    results_spec0["glm"]["lognormal"]["mu"]["mae"].append(mae(mu0, mu0_glm))
    results_spec0["glm"]["lognormal"]["sigma"]["mae"].append(mae(sigma0, sigma0_glm))
    results_spec0["glm"]["lognormal"]["mu"]["bias"].append(bias(mu0, mu0_glm))
    results_spec0["glm"]["lognormal"]["sigma"]["bias"].append(bias(sigma0, sigma0_glm))

    results_spec0["glm"]["gamma"]["nll"].append(-np.sum(scipy.stats.gamma.logpdf(x = gamma0, a = alpha0_glm*np.ones_like(theta0_glm), scale = theta0_glm)))
    results_spec0["glm"]["lognormal"]["nll"].append(-np.sum(scipy.stats.lognorm.logpdf(x = lognormal0, scale = np.exp(mu0_glm), s = sigma0_glm*np.ones_like(mu0_glm))))

    #MBT
    results_spec0["mbt"]["gamma"]["alpha"]["mse"].append(mse(alpha0, alpha0_mbt))
    results_spec0["mbt"]["gamma"]["theta"]["mse"].append(mse(theta0, theta0_mbt))
    results_spec0["mbt"]["gamma"]["alpha"]["mae"].append(mae(alpha0, alpha0_mbt))
    results_spec0["mbt"]["gamma"]["theta"]["mae"].append(mae(theta0, theta0_mbt))
    results_spec0["mbt"]["gamma"]["alpha"]["bias"].append(bias(alpha0, alpha0_mbt))
    results_spec0["mbt"]["gamma"]["theta"]["bias"].append(bias(theta0, theta0_mbt))
    
    results_spec0["mbt"]["lognormal"]["mu"]["mse"].append(mse(mu0, mu0_mbt))
    results_spec0["mbt"]["lognormal"]["sigma"]["mse"].append(mse(sigma0, sigma0_mbt))
    results_spec0["mbt"]["lognormal"]["mu"]["mae"].append(mae(mu0, mu0_mbt))
    results_spec0["mbt"]["lognormal"]["sigma"]["mae"].append(mae(sigma0, sigma0_mbt))
    results_spec0["mbt"]["lognormal"]["mu"]["bias"].append(bias(mu0, mu0_mbt))
    results_spec0["mbt"]["lognormal"]["sigma"]["bias"].append(bias(sigma0, sigma0_mbt))

    results_spec0["mbt"]["gamma"]["nll"].append(-np.sum(scipy.stats.gamma.logpdf(x = gamma0, a = alpha0_mbt, scale = theta0_mbt)))
    results_spec0["mbt"]["lognormal"]["nll"].append(-np.sum(scipy.stats.lognorm.logpdf(x = lognormal0, scale = np.exp(mu0_mbt), s = sigma0_mbt)))

    # simulate from scenario 2
    gamma2 = scipy.stats.gamma.rvs(a = alpha2, scale = theta2, loc = 0, size=(1,n)).reshape(-1,1)
    lognormal2 = scipy.stats.lognorm.rvs(s = sigma2, scale = np.exp(mu2), loc = 0, size=(1,n)).reshape(-1,1)

    # fit MBT and GLM models
    with suppress_stdout_stderr(): # suppress progress bar
        # gamma and lognormal MBT fit
        gamma2_mbt = MBT(n_boosts = 2000, learning_rate = 0.2, loss_type = "gamma", refit = False, min_leaf = 2500, lambda_weights=0.1).fit(x, gamma2, do_plot = False)
        lognormal2_mbt = MBT(n_boosts = 2000, learning_rate = 0.2, loss_type = "lognormal", refit = False, min_leaf = 2500, lambda_weights=0.1).fit(x, lognormal2, do_plot = False)
        # gamma and lognormal GLM fit
        gamma2_glm = sm.GLM(gamma2, x, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        #x_intercept = sm.add_constant(x)
        lognormal2_glm = sm.OLS(np.log(lognormal2), x_intercept).fit()

    # MBT predictions
    gamma2_mbt_pred = gamma2_mbt.predict(x)
    lognormal2_mbt_pred = lognormal2_mbt.predict(x)

    alpha2_mbt = np.exp(gamma2_mbt_pred[:,0]).reshape(-1,1)
    theta2_mbt = 1/np.exp(gamma2_mbt_pred[:,1]).reshape(-1,1)

    sigma2_mbt = np.sqrt(1/(2*np.exp(lognormal2_mbt_pred[:,1])).reshape(-1,1))
    mu2_mbt = lognormal2_mbt_pred[:,0].reshape(-1,1)*sigma2_mbt

    # GLM predictions
    alpha2_glm = 1/gamma2_glm.scale # scalar
    theta2_glm = (gamma2_glm.fittedvalues*gamma2_glm.scale).reshape(-1,1) # vector

    mu2_glm = lognormal2_glm.fittedvalues.reshape(-1,1) # vector
    sigma2_glm = np.sqrt(lognormal2_glm.scale) # scalar

    # fit MLE (intercept-only) model
    alpha2_mle, _, theta2_mle = scipy.stats.gamma.fit(gamma2, floc=0)
    sigma2_mle, _, mu2_mle = scipy.stats.lognorm.fit(lognormal2, floc = 0)
    mu2_mle = np.log(mu2_mle)

    # performance measures
    # MLE
    results_spec2["mle"]["gamma"]["alpha"]["mse"].append(mse(alpha2, alpha2_mle))
    results_spec2["mle"]["gamma"]["theta"]["mse"].append(mse(theta2, theta2_mle))
    results_spec2["mle"]["gamma"]["alpha"]["mae"].append(mae(alpha2, alpha2_mle))
    results_spec2["mle"]["gamma"]["theta"]["mae"].append(mae(theta2, theta2_mle))
    results_spec2["mle"]["gamma"]["alpha"]["bias"].append(bias(alpha2, alpha2_mle))
    results_spec2["mle"]["gamma"]["theta"]["bias"].append(bias(theta2, theta2_mle))
    
    results_spec2["mle"]["lognormal"]["mu"]["mse"].append(mse(mu2, mu2_mle))
    results_spec2["mle"]["lognormal"]["sigma"]["mse"].append(mse(sigma2, sigma2_mle))
    results_spec2["mle"]["lognormal"]["mu"]["mae"].append(mae(mu2, mu2_mle))
    results_spec2["mle"]["lognormal"]["sigma"]["mae"].append(mae(sigma2, sigma2_mle))
    results_spec2["mle"]["lognormal"]["mu"]["bias"].append(bias(mu2, mu2_mle))
    results_spec2["mle"]["lognormal"]["sigma"]["bias"].append(bias(sigma2, sigma2_mle))

    results_spec2["mle"]["gamma"]["nll"].append(-np.sum(scipy.stats.gamma.logpdf(x = gamma2, a = alpha2_mle, scale = theta2_mle)))
    results_spec2["mle"]["lognormal"]["nll"].append(-np.sum(scipy.stats.lognorm.logpdf(x = lognormal2, scale = np.exp(mu2_mle), s = sigma2_mle)))

    #GLM
    results_spec2["glm"]["gamma"]["alpha"]["mse"].append(mse(alpha2, alpha2_glm))
    results_spec2["glm"]["gamma"]["theta"]["mse"].append(mse(theta2, theta2_glm))
    results_spec2["glm"]["gamma"]["alpha"]["mae"].append(mae(alpha2, alpha2_glm))
    results_spec2["glm"]["gamma"]["theta"]["mae"].append(mae(theta2, theta2_glm))
    results_spec2["glm"]["gamma"]["alpha"]["bias"].append(bias(alpha2, alpha2_glm))
    results_spec2["glm"]["gamma"]["theta"]["bias"].append(bias(theta2, theta2_glm))
    
    results_spec2["glm"]["lognormal"]["mu"]["mse"].append(mse(mu2, mu2_glm))
    results_spec2["glm"]["lognormal"]["sigma"]["mse"].append(mse(sigma2, sigma2_glm))
    results_spec2["glm"]["lognormal"]["mu"]["mae"].append(mae(mu2, mu2_glm))
    results_spec2["glm"]["lognormal"]["sigma"]["mae"].append(mae(sigma2, sigma2_glm))
    results_spec2["glm"]["lognormal"]["mu"]["bias"].append(bias(mu2, mu2_glm))
    results_spec2["glm"]["lognormal"]["sigma"]["bias"].append(bias(sigma2, sigma2_glm))

    results_spec2["glm"]["gamma"]["nll"].append(-np.sum(scipy.stats.gamma.logpdf(x = gamma2, a = alpha2_glm*np.ones_like(theta2_glm), scale = theta2_glm)))
    results_spec2["glm"]["lognormal"]["nll"].append(-np.sum(scipy.stats.lognorm.logpdf(x = lognormal2, scale = np.exp(mu2_glm), s = sigma2_glm*np.ones_like(mu2_glm))))

    #MBT
    results_spec2["mbt"]["gamma"]["alpha"]["mse"].append(mse(alpha2, alpha2_mbt))
    results_spec2["mbt"]["gamma"]["theta"]["mse"].append(mse(theta2, theta2_mbt))
    results_spec2["mbt"]["gamma"]["alpha"]["mae"].append(mae(alpha2, alpha2_mbt))
    results_spec2["mbt"]["gamma"]["theta"]["mae"].append(mae(theta2, theta2_mbt))
    results_spec2["mbt"]["gamma"]["alpha"]["bias"].append(bias(alpha2, alpha2_mbt))
    results_spec2["mbt"]["gamma"]["theta"]["bias"].append(bias(theta2, theta2_mbt))
    
    results_spec2["mbt"]["lognormal"]["mu"]["mse"].append(mse(mu2, mu2_mbt))
    results_spec2["mbt"]["lognormal"]["sigma"]["mse"].append(mse(sigma2, sigma2_mbt))
    results_spec2["mbt"]["lognormal"]["mu"]["mae"].append(mae(mu2, mu2_mbt))
    results_spec2["mbt"]["lognormal"]["sigma"]["mae"].append(mae(sigma2, sigma2_mbt))
    results_spec2["mbt"]["lognormal"]["mu"]["bias"].append(bias(mu2, mu2_mbt))
    results_spec2["mbt"]["lognormal"]["sigma"]["bias"].append(bias(sigma2, sigma2_mbt))

    results_spec2["mbt"]["gamma"]["nll"].append(-np.sum(scipy.stats.gamma.logpdf(x = gamma2, a = alpha2_mbt, scale = theta2_mbt)))
    results_spec2["mbt"]["lognormal"]["nll"].append(-np.sum(scipy.stats.lognorm.logpdf(x = lognormal2, scale = np.exp(mu2_mbt), s = sigma2_mbt)))

# write to json file
with open('marginal_results_spec0.json', 'w', encoding='utf-8') as f:
    json.dump(results_spec0, f, ensure_ascii=False, indent=4)

with open('marginal_results_spec2.json', 'w', encoding='utf-8') as f:
    json.dump(results_spec2, f, ensure_ascii=False, indent=4)
