import numpy as np
import scipy.special as special
from scipy.stats import gamma, lognorm
from xgboost import XGBRegressor
from sklearn.linear_model import GammaRegressor, LinearRegression
from sklearn.metrics import mean_gamma_deviance
import sys
import os
from contextlib import contextmanager
import json

# Set project root relative to notebook location
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Losses import gaussian_xgb_loss as gauss
from Losses import clayton_xgb_loss as clayton
from Losses import gumbel_xgb_loss as gumbel

from sklearn.model_selection import train_test_split

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def mse(y, y_pred):
    return np.sum((y-y_pred)**2)

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

def copula_to_response(copula_samples, mu1, mu2):
    """
    Transform copula samples to lognormal margins with variable location parameters.
    
    Parameters:
        copula_samples (np.ndarray): shape (n, 2), values in (0, 1)
        mu1 (np.ndarray): mean (n,) parameters for Y1
        mu2 (np.ndarray): mean (n,), rate parameters for Y2
    
    Returns:
        np.ndarray: shape (n, 2), dependent lognormal samples
    """
    u1, u2 = copula_samples[:, 0], copula_samples[:, 1]
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)

    if np.any(u1 <= 0) or np.any(u2 <= 0):
        raise ValueError("Copula samples must be in (0, 1), not inclusive.")
    if np.any(mu1 <= 0) or np.any(mu2 <= 0):
        raise ValueError("Rate parameters must be > 0.")

    y1 = lognorm.ppf(u1, s = 1.5, scale=mu1)
    y2 = gamma.ppf(u2, a = 1.5, scale = mu2/1.5) 

    return np.column_stack((y1, y2))

def partial_dependence_with_truth(
    model,
    X,
    feature_index,
    true_theta_fn=None,
    transform=None,
    grid=None,
    grid_resolution=50
):
    X = np.asarray(X)
    
    # Grid
    if grid is None:
        xj = X[:, feature_index]
        grid = np.linspace(xj.min(), xj.max(), grid_resolution)
    
    pdp = np.zeros_like(grid)
    truth = np.zeros_like(grid) if true_theta_fn else None
    
    for i, val in enumerate(grid):
        X_temp = X.copy()
        X_temp[:, feature_index] = val
        
        # Model PDP
        preds = model.predict(X_temp)
        if transform:
            preds = transform(preds)
        pdp[i] = preds.mean()
        
        # True PDP
        if true_theta_fn:
            theta_true = true_theta_fn(X_temp)
            #if transform:
                #theta_true = transform(theta_true)
            truth[i] = theta_true.mean()
    
    return grid, pdp.tolist(), truth

gamma_model2 = XGBRegressor(
    objective="reg:gamma",
    n_estimators=500,
    tree_method="hist",
    eval_metric="gamma-nloglik",
    early_stopping_rounds = 3,
    learning_rate = 0.01,
    reg_lambda = 0.0,
    max_depth = 1,
    subsample = 1.0,
    colsample_bytree = 1.0,
    importance_type="total_gain"
)

normal_model1 = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds = 3,
    learning_rate = 0.01,
    reg_lambda = 0.0,
    max_depth = 1,
    subsample = 1.0,
    colsample_bytree = 1.0,
    importance_type="total_gain"
)

gauss_model = XGBRegressor(
    objective=gauss.gaussian_copula_nll_obj,
    n_estimators=500,
    tree_method="hist",
    eval_metric=gauss.gaussian_copula_nll_metric,
    early_stopping_rounds=3,
    learning_rate=0.01,
    reg_lambda=0.0,
    max_depth=2,
    subsample=1.0,
    colsample_bytree=1.0,
    importance_type="total_gain"
)

clayton_model = XGBRegressor(
    objective=clayton.clayton_copula_nll_obj,
    n_estimators=500,
    tree_method="hist",
    eval_metric=clayton.clayton_copula_nll_metric,
    early_stopping_rounds=3,
    learning_rate=0.01,
    reg_lambda=0.2,
    max_depth=1,
    subsample=1.0,
    colsample_bytree=1.0,
    importance_type="total_gain"
)

gumbel_model = XGBRegressor(
    objective=gumbel.gumbel_copula_nll_obj,
    n_estimators=500,
    tree_method="hist",
    eval_metric=gumbel.gumbel_copula_nll_metric,
    early_stopping_rounds=3,
    learning_rate=0.01,
    reg_lambda=0.2,
    max_depth=2,
    subsample=1.0,
    colsample_bytree=1.0,
    importance_type="total_gain"
)

gamma_bench_model = GammaRegressor(alpha=0.0, max_iter=500)
normal_bench_model = LinearRegression()

np.random.seed(seed=321)
x_pdp = np.random.uniform(-1, 1, (5000, 10))

# shape of generated data.
n_obs = 2000
p = 10

grid_resolution = 50

n_sim = 1

results = {
    "gaussian": {"lognormal": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "deviance_xgb" : [], 
               "deviance_glm": [],
               "mse_xgb": [],
               "mse_glm": []
               },
               "gamma": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "deviance_xgb" : [],
               "deviance_glm": [],
               "mse_xgb": [],
               "mse_glm": []
               },
               "pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "nll_xgb" : [], 
               "nll_constant": [],
               "mse_xgb": [],
               "mse_constant": []
               },
    "clayton": {"lognormal": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "deviance_xgb" : [], 
               "deviance_glm": [],
               "mse_xgb": [],
               "mse_glm": []
               },
               "gamma": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "deviance_xgb" : [],
               "deviance_glm": [],
               "mse_xgb": [],
               "mse_glm": []
               },
               "pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "nll_xgb" : [], 
               "nll_constant": [],
               "mse_xgb": [],
               "mse_constant": []
               },
    "gumbel": {"lognormal": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "deviance_xgb" : [], 
               "deviance_glm": [],
               "mse_xgb": [],
               "mse_glm": []
               },
               "gamma": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "deviance_xgb" : [],
               "deviance_glm": [],
               "mse_xgb": [],
               "mse_glm": []
               },
               "pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x2" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x3" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
                        "x4" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)]
                        }, 
               "feature_importance" : [[None for _ in range(p)] for _ in range(n_sim)], 
               "nll_xgb" : [], 
               "nll_constant": [],
               "mse_xgb": [],
               "mse_constant": []
               }
}

def true_mu1(X):
    x1 = X[:, 0]
    x3 = X[:, 2]
        
    return np.exp(-0.75*x1 + 0.5*np.cos(np.pi*x3))
    
def true_mu2(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
        
    return np.exp(0.5 - 0.7*x1 - 0.02*np.exp(2*(x2 + 1)))
    
def true_tau(X):
    x4 = X[:, 3]

    return special.expit(-0.8 + 1.5*np.log(4.5 - 1.7*np.sin(np.pi*x4)))

for i in range(n_sim):
    
    seed = i+1
    np.random.seed(seed=seed)

    x = np.random.uniform(-1, 1, (n_obs, p))
    
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    
    mu1 = true_mu1(x)
    mu2 = true_mu2(x)
    tau = true_tau(x)

    # Gaussian copula
    u_gauss = clip_copula_samples(gauss.sample_gaussian_copula(gauss.gauss_tau2par(tau), seed=seed), eps=1e-8)

    # Generate dependent bivariate gamma observations from gaussian copula observations
    y_gauss = copula_to_response(u_gauss, mu1, mu2)

    #train, validation and test data split (70,10,20)
    x_main_gauss, x_test_gauss, y_main_gauss, y_test_gauss = train_test_split(x, y_gauss, test_size=0.2, random_state=42)
    x_train_gauss, x_val_gauss, y_train_gauss, y_val_gauss = train_test_split(x_main_gauss, y_main_gauss, test_size=0.125, random_state=42)

    # Fitting marginal distributions
    normal_model1.set_params(learning_rate = 0.01, max_depth = 1, reg_lambda = 0.0)
    gamma_model2.set_params(learning_rate = 0.01, max_depth = 1, reg_lambda = 0.0)

    with suppress_stdout_stderr(): # suppress progress bar
        normal_xgb1 = normal_model1.fit(x_train_gauss, np.log(y_train_gauss[:,0]), eval_set=[(x_val_gauss, np.log(y_val_gauss[:,0]))])
        gamma_xgb2 = gamma_model2.fit(x_train_gauss, y_train_gauss[:,1], eval_set=[(x_val_gauss, y_val_gauss[:,1])])

        normal_glm = normal_bench_model.fit(x_train_gauss, np.log(y_train_gauss[:,0]))
        gamma_glm = gamma_bench_model.fit(x_train_gauss, y_train_gauss[:,1])

    _, pdp_normal_x1, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x2, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x3, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x4, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x1, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x2, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x3, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x4, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )
    
    mu1_train_hat = normal_xgb1.predict(x_train_gauss)
    mu2_train_hat = gamma_xgb2.predict(x_train_gauss)

    mu1_val_hat = normal_xgb1.predict(x_val_gauss)
    mu2_val_hat = gamma_xgb2.predict(x_val_gauss)

    mu1_test_hat = normal_xgb1.predict(x_test_gauss)
    mu2_test_hat = gamma_xgb2.predict(x_test_gauss)

    results["gaussian"]["lognormal"]["pdp"]["x1"][i] = pdp_normal_x1
    results["gaussian"]["lognormal"]["pdp"]["x2"][i] = pdp_normal_x2
    results["gaussian"]["lognormal"]["pdp"]["x3"][i] = pdp_normal_x3
    results["gaussian"]["lognormal"]["pdp"]["x4"][i] = pdp_normal_x4
    results["gaussian"]["lognormal"]["feature_importance"][i] = normal_xgb1.feature_importances_.tolist()
    results["gaussian"]["lognormal"]["deviance_xgb"].append(mse(np.log(y_test_gauss[:,0]), normal_xgb1.predict(x_test_gauss)))
    results["gaussian"]["lognormal"]["deviance_glm"].append(mse(np.log(y_test_gauss[:,0]), normal_glm.predict(x_test_gauss)))
    results["gaussian"]["lognormal"]["mse_xgb"].append(mse(true_mu1(x_test_gauss), np.exp(mu1_test_hat)))
    results["gaussian"]["lognormal"]["mse_glm"].append(mse(true_mu1(x_test_gauss), np.exp(normal_glm.predict(x_test_gauss))))

    results["gaussian"]["gamma"]["pdp"]["x1"][i] = pdp_gamma_x1
    results["gaussian"]["gamma"]["pdp"]["x2"][i] = pdp_gamma_x2
    results["gaussian"]["gamma"]["pdp"]["x3"][i] = pdp_gamma_x3
    results["gaussian"]["gamma"]["pdp"]["x4"][i] = pdp_gamma_x4
    results["gaussian"]["gamma"]["feature_importance"][i] = gamma_xgb2.feature_importances_.tolist()
    results["gaussian"]["gamma"]["deviance_xgb"].append(mean_gamma_deviance(y_test_gauss[:,1], gamma_xgb2.predict(x_test_gauss)))
    results["gaussian"]["gamma"]["deviance_glm"].append(mean_gamma_deviance(y_test_gauss[:,1], gamma_glm.predict(x_test_gauss)))
    results["gaussian"]["gamma"]["mse_xgb"].append(mse(true_mu2(x_test_gauss), mu2_test_hat))
    results["gaussian"]["gamma"]["mse_glm"].append(mse(true_mu2(x_test_gauss), gamma_glm.predict(x_test_gauss)))

    pseudo_obs_train_gauss = clip_copula_samples(np.column_stack((lognorm.cdf(y_train_gauss[:, 0], s = 1.5, scale=np.exp(mu1_train_hat)), gamma.cdf(y_train_gauss[:, 1], a=1.5, scale=mu2_train_hat/1.5))), eps=1e-8)
    pseudo_obs_val_gauss = clip_copula_samples(np.column_stack((lognorm.cdf(y_val_gauss[:, 0], s = 1.5, scale=np.exp(mu1_val_hat)), gamma.cdf(y_val_gauss[:, 1], a=1.5, scale=mu2_val_hat/1.5))), eps=1e-8)
    pseudo_obs_test_gauss = clip_copula_samples(np.column_stack((lognorm.cdf(y_test_gauss[:, 0], s = 1.5, scale=np.exp(mu1_test_hat)), gamma.cdf(y_test_gauss[:, 1], a=1.5, scale=mu2_test_hat/1.5))), eps=1e-8)

    # Fitting XGBoost copula model to pseudo-observations
    try:
        with suppress_stdout_stderr(): # suppress progress bar
            gauss_xgb = gauss_model.fit(x_train_gauss, pseudo_obs_train_gauss, eval_set=[(x_val_gauss, pseudo_obs_val_gauss)])

        test_preds_gauss = gauss_xgb.predict(x_test_gauss)[:,0].reshape(-1,1)

        _, pdp_gauss_x1, _ = partial_dependence_with_truth(
            gauss_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau,
            transform=gauss.gauss_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_gauss_x2, _ = partial_dependence_with_truth(
            gauss_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau,
            transform=gauss.gauss_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_gauss_x3, _ = partial_dependence_with_truth(
            gauss_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau,
            transform=gauss.gauss_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_gauss_x4, _ = partial_dependence_with_truth(
            gauss_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau,
            transform=gauss.gauss_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        results["gaussian"]["pdp"]["x1"][i] = pdp_gauss_x1
        results["gaussian"]["pdp"]["x2"][i] = pdp_gauss_x2
        results["gaussian"]["pdp"]["x3"][i] = pdp_gauss_x3
        results["gaussian"]["pdp"]["x4"][i] = pdp_gauss_x4
        results["gaussian"]["feature_importance"][i] = gauss_xgb.feature_importances_.tolist()
        results["gaussian"]["nll_xgb"].append(gauss.gaussian_copula_nll_metric(pseudo_obs_test_gauss, test_preds_gauss))
        results["gaussian"]["nll_constant"].append(gauss.gaussian_copula_nll_metric(pseudo_obs_test_gauss, gauss.get_initial_guess(pseudo_obs_test_gauss)))
        results["gaussian"]["mse_xgb"].append(mse(true_tau(x_test_gauss).reshape(-1,1), gauss.gauss_predt2tau(test_preds_gauss)))
        results["gaussian"]["mse_constant"].append(mse(true_tau(x_test_gauss), gauss.gauss_predt2tau(gauss.get_initial_guess(pseudo_obs_test_gauss))))
    except:
        pass


    # Clayton copula
    u_clayton = clip_copula_samples(clayton.sample_clayton_copula(clayton.clayton_tau2par(tau), seed=seed), eps=1e-8)

    # Generate dependent bivariate gamma observations from clayton copula observations
    y_clayton = copula_to_response(u_clayton, mu1, mu2)

    #train, validation and test data split (70,10,20)
    x_main_clayton, x_test_clayton, y_main_clayton, y_test_clayton = train_test_split(x, y_clayton, test_size=0.2, random_state=42)
    x_train_clayton, x_val_clayton, y_train_clayton, y_val_clayton = train_test_split(x_main_clayton, y_main_clayton, test_size=0.125, random_state=42)

    # Fitting marginal distributions
    normal_model1.set_params(learning_rate = 0.01, max_depth = 1, reg_lambda = 0.0)
    gamma_model2.set_params(learning_rate = 0.01, max_depth = 1, reg_lambda = 0.0)
    with suppress_stdout_stderr(): # suppress progress bar
        normal_xgb1 = normal_model1.fit(x_train_clayton, np.log(y_train_clayton[:,0]), eval_set=[(x_val_clayton, np.log(y_val_clayton[:,0]))])
        gamma_xgb2 = gamma_model2.fit(x_train_clayton, y_train_clayton[:,1], eval_set=[(x_val_clayton, y_val_clayton[:,1])])

        normal_glm = normal_bench_model.fit(x_train_clayton, np.log(y_train_clayton[:,0]))
        gamma_glm = gamma_bench_model.fit(x_train_clayton, y_train_clayton[:,1])

    _, pdp_normal_x1, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x2, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x3, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x4, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x1, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x2, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x3, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x4, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )
    
    mu1_train_hat = normal_xgb1.predict(x_train_clayton)
    mu2_train_hat = gamma_xgb2.predict(x_train_clayton)

    mu1_val_hat = normal_xgb1.predict(x_val_clayton)
    mu2_val_hat = gamma_xgb2.predict(x_val_clayton)

    mu1_test_hat = normal_xgb1.predict(x_test_clayton)
    mu2_test_hat = gamma_xgb2.predict(x_test_clayton)

    results["clayton"]["lognormal"]["pdp"]["x1"][i] = pdp_normal_x1
    results["clayton"]["lognormal"]["pdp"]["x2"][i] = pdp_normal_x2
    results["clayton"]["lognormal"]["pdp"]["x3"][i] = pdp_normal_x3
    results["clayton"]["lognormal"]["pdp"]["x4"][i] = pdp_normal_x4
    results["clayton"]["lognormal"]["feature_importance"][i] = normal_xgb1.feature_importances_.tolist()
    results["clayton"]["lognormal"]["deviance_xgb"].append(mse(np.log(y_test_clayton[:,0]), normal_xgb1.predict(x_test_clayton)))
    results["clayton"]["lognormal"]["deviance_glm"].append(mse(np.log(y_test_clayton[:,0]), normal_glm.predict(x_test_clayton)))
    results["clayton"]["lognormal"]["mse_xgb"].append(mse(true_mu1(x_test_clayton), np.exp(mu1_test_hat)))
    results["clayton"]["lognormal"]["mse_glm"].append(mse(true_mu1(x_test_clayton), np.exp(normal_glm.predict(x_test_clayton))))

    results["clayton"]["gamma"]["pdp"]["x1"][i] = pdp_gamma_x1
    results["clayton"]["gamma"]["pdp"]["x2"][i] = pdp_gamma_x2
    results["clayton"]["gamma"]["pdp"]["x3"][i] = pdp_gamma_x3
    results["clayton"]["gamma"]["pdp"]["x4"][i] = pdp_gamma_x4
    results["clayton"]["gamma"]["feature_importance"][i] = gamma_xgb2.feature_importances_.tolist()
    results["clayton"]["gamma"]["deviance_xgb"].append(mean_gamma_deviance(y_test_clayton[:,1], gamma_xgb2.predict(x_test_clayton)))
    results["clayton"]["gamma"]["deviance_glm"].append(mean_gamma_deviance(y_test_clayton[:,1], gamma_glm.predict(x_test_clayton)))
    results["clayton"]["gamma"]["mse_xgb"].append(mse(true_mu2(x_test_clayton), mu2_test_hat))
    results["clayton"]["gamma"]["mse_glm"].append(mse(true_mu2(x_test_clayton), gamma_glm.predict(x_test_clayton)))

    pseudo_obs_train_clayton = clip_copula_samples(np.column_stack((lognorm.cdf(y_train_clayton[:, 0], s = 1.5, scale=np.exp(mu1_train_hat)), gamma.cdf(y_train_clayton[:, 1], a=1.5, scale=mu2_train_hat/1.5))), eps=1e-8)
    pseudo_obs_val_clayton = clip_copula_samples(np.column_stack((lognorm.cdf(y_val_clayton[:, 0], s = 1.5, scale=np.exp(mu1_val_hat)), gamma.cdf(y_val_clayton[:, 1], a=1.5, scale=mu2_val_hat/1.5))), eps=1e-8)
    pseudo_obs_test_clayton = clip_copula_samples(np.column_stack((lognorm.cdf(y_test_clayton[:, 0], s = 1.5, scale=np.exp(mu1_test_hat)), gamma.cdf(y_test_clayton[:, 1], a=1.5, scale=mu2_test_hat/1.5))), eps=1e-8)

    # Fitting XGBoost copula model to pseudo-observations
    try:
        with suppress_stdout_stderr(): # suppress progress bar
            clayton_xgb = clayton_model.fit(x_train_clayton, pseudo_obs_train_clayton, eval_set=[(x_val_clayton, pseudo_obs_val_clayton)])

        test_preds_clayton = clayton_xgb.predict(x_test_clayton)[:,0].reshape(-1,1)

        _, pdp_clayton_x1, _ = partial_dependence_with_truth(
            clayton_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_clayton_x2, _ = partial_dependence_with_truth(
            clayton_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_clayton_x3, _ = partial_dependence_with_truth(
            clayton_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_clayton_x4, _ = partial_dependence_with_truth(
            clayton_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        results["clayton"]["pdp"]["x1"][i] = pdp_clayton_x1
        results["clayton"]["pdp"]["x2"][i] = pdp_clayton_x2
        results["clayton"]["pdp"]["x3"][i] = pdp_clayton_x3
        results["clayton"]["pdp"]["x4"][i] = pdp_clayton_x4
        results["clayton"]["feature_importance"][i] = clayton_xgb.feature_importances_.tolist()
        results["clayton"]["nll_xgb"].append(clayton.clayton_copula_nll_metric(pseudo_obs_test_clayton, test_preds_clayton))
        results["clayton"]["nll_constant"].append(clayton.clayton_copula_nll_metric(pseudo_obs_test_clayton, clayton.get_initial_guess(pseudo_obs_test_clayton)))
        results["clayton"]["mse_xgb"].append(mse(true_tau(x_test_clayton).reshape(-1,1), clayton.clayton_predt2tau(test_preds_clayton)))
        results["clayton"]["mse_constant"].append(mse(true_tau(x_test_clayton), clayton.clayton_predt2tau(clayton.get_initial_guess(pseudo_obs_test_clayton))))
    except:
        pass


    # Gumbel copula
    u_gumbel = clip_copula_samples(gumbel.sample_gumbel_copula(gumbel.gumbel_tau2par(tau), seed=seed), eps=1e-8)

    # Generate dependent bivariate gamma observations from gumbel copula observations
    y_gumbel = copula_to_response(u_gumbel, mu1, mu2)

    #train, validation and test data split (70,10,20)
    x_main_gumbel, x_test_gumbel, y_main_gumbel, y_test_gumbel = train_test_split(x, y_gumbel, test_size=0.2, random_state=42)
    x_train_gumbel, x_val_gumbel, y_train_gumbel, y_val_gumbel = train_test_split(x_main_gumbel, y_main_gumbel, test_size=0.125, random_state=42)

    # Fitting marginal distributions
    normal_model1.set_params(learning_rate = 0.01, max_depth = 2, reg_lambda = 0.0)
    gamma_model2.set_params(learning_rate = 0.01, max_depth = 1, reg_lambda = 0.0)

    with suppress_stdout_stderr(): # suppress progress bar
        normal_xgb1 = normal_model1.fit(x_train_gumbel, np.log(y_train_gumbel[:,0]), eval_set=[(x_val_gumbel, np.log(y_val_gumbel[:,0]))])
        gamma_xgb2 = gamma_model2.fit(x_train_gumbel, y_train_gumbel[:,1], eval_set=[(x_val_gumbel, y_val_gumbel[:,1])])

        normal_glm = normal_bench_model.fit(x_train_gumbel, np.log(y_train_gumbel[:,0]))
        gamma_glm = gamma_bench_model.fit(x_train_gumbel, y_train_gumbel[:,1])

    _, pdp_normal_x1, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x2, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x3, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_normal_x4, _ = partial_dependence_with_truth(
        normal_xgb1,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_mu1,
        transform=np.exp,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x1, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x2, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x3, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )

    _, pdp_gamma_x4, _ = partial_dependence_with_truth(
        gamma_xgb2,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_mu2,
        transform=None,
        grid = np.linspace(-1,1,grid_resolution)
    )
    
    mu1_train_hat = normal_xgb1.predict(x_train_gumbel)
    mu2_train_hat = gamma_xgb2.predict(x_train_gumbel)

    mu1_val_hat = normal_xgb1.predict(x_val_gumbel)
    mu2_val_hat = gamma_xgb2.predict(x_val_gumbel)

    mu1_test_hat = normal_xgb1.predict(x_test_gumbel)
    mu2_test_hat = gamma_xgb2.predict(x_test_gumbel)

    results["gumbel"]["lognormal"]["pdp"]["x1"][i] = pdp_normal_x1
    results["gumbel"]["lognormal"]["pdp"]["x2"][i] = pdp_normal_x2
    results["gumbel"]["lognormal"]["pdp"]["x3"][i] = pdp_normal_x3
    results["gumbel"]["lognormal"]["pdp"]["x4"][i] = pdp_normal_x4
    results["gumbel"]["lognormal"]["feature_importance"][i] = normal_xgb1.feature_importances_.tolist()
    results["gumbel"]["lognormal"]["deviance_xgb"].append(mse(np.log(y_test_gumbel[:,0]), normal_xgb1.predict(x_test_gumbel)))
    results["gumbel"]["lognormal"]["deviance_glm"].append(mse(np.log(y_test_gumbel[:,0]), normal_glm.predict(x_test_gumbel)))
    results["gumbel"]["lognormal"]["mse_xgb"].append(mse(true_mu1(x_test_gumbel), np.exp(mu1_test_hat)))
    results["gumbel"]["lognormal"]["mse_glm"].append(mse(true_mu1(x_test_gumbel), np.exp(normal_glm.predict(x_test_gumbel))))

    results["gumbel"]["gamma"]["pdp"]["x1"][i] = pdp_gamma_x1
    results["gumbel"]["gamma"]["pdp"]["x2"][i] = pdp_gamma_x2
    results["gumbel"]["gamma"]["pdp"]["x3"][i] = pdp_gamma_x3
    results["gumbel"]["gamma"]["pdp"]["x4"][i] = pdp_gamma_x4
    results["gumbel"]["gamma"]["feature_importance"][i] = gamma_xgb2.feature_importances_.tolist()
    results["gumbel"]["gamma"]["deviance_xgb"].append(mean_gamma_deviance(y_test_gumbel[:,1], gamma_xgb2.predict(x_test_gumbel)))
    results["gumbel"]["gamma"]["deviance_glm"].append(mean_gamma_deviance(y_test_gumbel[:,1], gamma_glm.predict(x_test_gumbel)))
    results["gumbel"]["gamma"]["mse_xgb"].append(mse(true_mu2(x_test_gumbel), mu2_test_hat))
    results["gumbel"]["gamma"]["mse_glm"].append(mse(true_mu2(x_test_gumbel), gamma_glm.predict(x_test_gumbel)))

    pseudo_obs_train_gumbel = clip_copula_samples(np.column_stack((lognorm.cdf(y_train_gumbel[:, 0], s = 1.5, scale=np.exp(mu1_train_hat)), gamma.cdf(y_train_gumbel[:, 1], a=1.5, scale=mu2_train_hat/1.5))), eps=1e-8)
    pseudo_obs_val_gumbel = clip_copula_samples(np.column_stack((lognorm.cdf(y_val_gumbel[:, 0], s = 1.5, scale=np.exp(mu1_val_hat)), gamma.cdf(y_val_gumbel[:, 1], a=1.5, scale=mu2_val_hat/1.5))), eps=1e-8)
    pseudo_obs_test_gumbel = clip_copula_samples(np.column_stack((lognorm.cdf(y_test_gumbel[:, 0], s = 1.5, scale=np.exp(mu1_test_hat)), gamma.cdf(y_test_gumbel[:, 1], a=1.5, scale=mu2_test_hat/1.5))), eps=1e-8)

    # Fitting XGBoost copula model to pseudo-observations
    try:
        with suppress_stdout_stderr(): # suppress progress bar
            gumbel_xgb = gumbel_model.fit(x_train_gumbel, pseudo_obs_train_gumbel, eval_set=[(x_val_gumbel, pseudo_obs_val_gumbel)])

        test_preds_gumbel = gumbel_xgb.predict(x_test_gumbel)[:,0].reshape(-1,1)

        _, pdp_gumbel_x1, _ = partial_dependence_with_truth(
            gumbel_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_gumbel_x2, _ = partial_dependence_with_truth(
            gumbel_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_gumbel_x3, _ = partial_dependence_with_truth(
            gumbel_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        _, pdp_gumbel_x4, _ = partial_dependence_with_truth(
            gumbel_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(-1,1,grid_resolution)
        )

        results["gumbel"]["pdp"]["x1"][i] = pdp_gumbel_x1
        results["gumbel"]["pdp"]["x2"][i] = pdp_gumbel_x2
        results["gumbel"]["pdp"]["x3"][i] = pdp_gumbel_x3
        results["gumbel"]["pdp"]["x4"][i] = pdp_gumbel_x4
        results["gumbel"]["feature_importance"][i] = gumbel_xgb.feature_importances_.tolist()
        results["gumbel"]["nll_xgb"].append(gumbel.gumbel_copula_nll_metric(pseudo_obs_test_gumbel, test_preds_gumbel))
        results["gumbel"]["nll_constant"].append(gumbel.gumbel_copula_nll_metric(pseudo_obs_test_gumbel, gumbel.get_initial_guess(pseudo_obs_test_gumbel)))
        results["gumbel"]["mse_xgb"].append(mse(true_tau(x_test_gumbel).reshape(-1,1), gumbel.gumbel_predt2tau(test_preds_gumbel)))
        results["gumbel"]["mse_constant"].append(mse(true_tau(x_test_gumbel), gumbel.gumbel_predt2tau(gumbel.get_initial_guess(pseudo_obs_test_gumbel))))
    except:
        pass

# write to json file
with open('phase2_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)