import numpy as np
import scipy.special as special
from xgboost import XGBRegressor
import sys
import os
import json
from contextlib import contextmanager

# Set project root relative to notebook location
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Losses import gaussian_xgb_loss as gauss
from Losses import clayton_xgb_loss as clayton
from Losses import gumbel_xgb_loss as gumbel

from sklearn.model_selection import train_test_split

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

def partial_dependence_plot(
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
        
    return pdp.tolist()

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

gauss_model = XGBRegressor(
    objective=gauss.gaussian_copula_nll_obj,
    n_estimators=500,
    tree_method="hist",
    eval_metric=gauss.gaussian_copula_nll_metric,
    early_stopping_rounds=3,
    learning_rate=0.01,
    reg_lambda=0.2,
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
    learning_rate=0.1,
    reg_lambda=0.1,
    max_depth=1,
    subsample=1.0,
    colsample_bytree=1.0,
    importance_type="total_gain"
)

np.random.seed(seed=321)
x_pdp = np.random.uniform(0, 1, (5000, 10))

# shape of generated data.
n_obs = 2000
p = 10

#ratio = 0.75
#num_boost_round = 500

grid_resolution = 25

n_sim = 1

results = {
    "linear": {"gaussian": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
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
               "clayton": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
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
               "gumbel": {"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
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
               },
    "non-linear": {"gaussian":{"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
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
               "clayton":{"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
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
               "gumbel":{"pdp" : {"x1" : [[None for _ in range(grid_resolution)] for _ in range(n_sim)],
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
}

for i in range(n_sim):
    
    seed = i+1
    np.random.seed(seed=seed)

    x = np.random.uniform(0, 1, (n_obs, p))
    
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    
    # First linear scenario
    def true_tau1(X):
        x1 = X[:, 0]
        #x2 = X[:, 1]
        #x3 = X[:, 2]
        x4 = X[:, 3]

        return special.expit(1.2*x1 - 1.5*x4) #special.expit(0.5 - 0.3*x1 + 0.4*x2 - 2*x3)

    tau1 = true_tau1(x)

    # Second non-linear scenario
    def true_tau2(X):
        x1 = X[:, 0]
        #x2 = X[:, 1]
        #x3 = X[:, 2]
        x4 = X[:, 3]

        return special.expit(0.8*np.sin(2*np.pi*x1) + 1.5*(x1 > 0.5) - 1.2*(x4 > 0.7) + 0.6*x4**2) #special.expit(0.1 - 0.2*(x1 < 0.3) + 0.3*np.sin(2*np.pi*x2) + 0.5*x3**2 - 2.5*x3*(x2 > 0.5))

    tau2 = true_tau2(x)

    # Gaussian copula

    y_gauss1 = clip_copula_samples(gauss.sample_gaussian_copula(gauss.gauss_tau2par(tau1), seed=seed), eps=1e-7)

    #train, validation and test data split (70,10,20)
    x_main_gauss1, x_test_gauss1, y_main_gauss1, y_test_gauss1 = train_test_split(x, y_gauss1, test_size=0.2, random_state=42)
    x_train_gauss1, x_val_gauss1, y_train_gauss1, y_val_gauss1 = train_test_split(x_main_gauss1, y_main_gauss1, test_size=0.125, random_state=42)

    y_gauss2 = clip_copula_samples(gauss.sample_gaussian_copula(gauss.gauss_tau2par(tau2), seed=seed), eps=1e-7)

    #train, validation and test data split (70,10,20)
    x_main_gauss2, x_test_gauss2, y_main_gauss2, y_test_gauss2 = train_test_split(x, y_gauss2, test_size=0.2, random_state=42)
    x_train_gauss2, x_val_gauss2, y_train_gauss2, y_val_gauss2 = train_test_split(x_main_gauss2, y_main_gauss2, test_size=0.125, random_state=42)

    # Fitting gaussian copula to linear scenario
    gauss_model.set_params(learning_rate=0.01, reg_lambda=0.0, max_depth=2)
    with suppress_stdout_stderr(): # suppress progress bar
        gauss1_xgb = gauss_model.fit(x_train_gauss1, y_train_gauss1, eval_set=[(x_val_gauss1, y_val_gauss1)])

    test_preds_gauss1 = gauss1_xgb.predict(x_test_gauss1)[:,0].reshape(-1,1)

    pdp_gauss1_x1 = partial_dependence_plot(
        gauss1_xgb,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_tau1,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    pdp_gauss1_x2 = partial_dependence_plot(
        gauss1_xgb,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_tau1,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    pdp_gauss1_x3 = partial_dependence_plot(
        gauss1_xgb,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_tau1,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    pdp_gauss1_x4 = partial_dependence_plot(
        gauss1_xgb,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_tau1,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )


    results["linear"]["gaussian"]["pdp"]["x1"][i] = pdp_gauss1_x1
    results["linear"]["gaussian"]["pdp"]["x2"][i] = pdp_gauss1_x2
    results["linear"]["gaussian"]["pdp"]["x3"][i] = pdp_gauss1_x3
    results["linear"]["gaussian"]["pdp"]["x4"][i] = pdp_gauss1_x4

    results["linear"]["gaussian"]["feature_importance"][i] = gauss1_xgb.feature_importances_.tolist()

    results["linear"]["gaussian"]["nll_xgb"].append(gauss.gaussian_copula_nll_metric(y_test_gauss1, test_preds_gauss1))
    results["linear"]["gaussian"]["nll_constant"].append(gauss.gaussian_copula_nll_metric(y_test_gauss1, gauss.get_initial_guess(y_test_gauss1)))
    results["linear"]["gaussian"]["mse_xgb"].append(mse(true_tau1(x_test_gauss1).reshape(-1,1), gauss.gauss_predt2tau(test_preds_gauss1)))
    results["linear"]["gaussian"]["mse_constant"].append(mse(true_tau1(x_test_gauss1), gauss.gauss_predt2tau(gauss.get_initial_guess(y_test_gauss1))))


    # Fitting gaussian copula to non-linear scenario
    gauss_model.set_params(learning_rate=0.01, reg_lambda=0.0, max_depth=2)
    with suppress_stdout_stderr(): # suppress progress bar
        gauss2_xgb = gauss_model.fit(x_train_gauss2, y_train_gauss2, eval_set=[(x_val_gauss2, y_val_gauss2)])

    test_preds_gauss2 = gauss2_xgb.predict(x_test_gauss2)[:,0].reshape(-1,1)

    pdp_gauss2_x1 = partial_dependence_plot(
        gauss2_xgb,
        x_pdp,
        feature_index=0,
        true_theta_fn=true_tau2,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    pdp_gauss2_x2 = partial_dependence_plot(
        gauss2_xgb,
        x_pdp,
        feature_index=1,
        true_theta_fn=true_tau2,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    pdp_gauss2_x3 = partial_dependence_plot(
        gauss2_xgb,
        x_pdp,
        feature_index=2,
        true_theta_fn=true_tau2,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    pdp_gauss2_x4 = partial_dependence_plot(
        gauss2_xgb,
        x_pdp,
        feature_index=3,
        true_theta_fn=true_tau2,
        transform=gauss.gauss_predt2tau,
        grid = np.linspace(0,1,grid_resolution)
    )

    results["non-linear"]["gaussian"]["pdp"]["x1"][i] = pdp_gauss2_x1
    results["non-linear"]["gaussian"]["pdp"]["x2"][i] = pdp_gauss2_x2
    results["non-linear"]["gaussian"]["pdp"]["x3"][i] = pdp_gauss2_x3
    results["non-linear"]["gaussian"]["pdp"]["x4"][i] = pdp_gauss2_x4

    results["non-linear"]["gaussian"]["feature_importance"][i] = gauss2_xgb.feature_importances_.tolist()

    results["non-linear"]["gaussian"]["nll_xgb"].append(gauss.gaussian_copula_nll_metric(y_test_gauss2, test_preds_gauss2))
    results["non-linear"]["gaussian"]["nll_constant"].append(gauss.gaussian_copula_nll_metric(y_test_gauss2, gauss.get_initial_guess(y_test_gauss2)))
    results["non-linear"]["gaussian"]["mse_xgb"].append(mse(true_tau2(x_test_gauss2).reshape(-1,1), gauss.gauss_predt2tau(test_preds_gauss2)))
    results["non-linear"]["gaussian"]["mse_constant"].append(mse(true_tau2(x_test_gauss2), gauss.gauss_predt2tau(gauss.get_initial_guess(y_test_gauss2))))
    

    #Clayton copula

    y_clayton1 = clip_copula_samples(clayton.sample_clayton_copula(clayton.clayton_tau2par(tau1), seed=seed), eps=1e-7)

    #train, validation and test data split (70,10,20)
    x_main_clayton1, x_test_clayton1, y_main_clayton1, y_test_clayton1 = train_test_split(x, y_clayton1, test_size=0.2, random_state=42)
    x_train_clayton1, x_val_clayton1, y_train_clayton1, y_val_clayton1 = train_test_split(x_main_clayton1, y_main_clayton1, test_size=0.125, random_state=42)

    y_clayton2 = clip_copula_samples(clayton.sample_clayton_copula(clayton.clayton_tau2par(tau2), seed=seed), eps=1e-7)

    #train, validation and test data split (70,10,20)
    x_main_clayton2, x_test_clayton2, y_main_clayton2, y_test_clayton2 = train_test_split(x, y_clayton2, test_size=0.2, random_state=42)
    x_train_clayton2, x_val_clayton2, y_train_clayton2, y_val_clayton2 = train_test_split(x_main_clayton2, y_main_clayton2, test_size=0.125, random_state=42)

    # Fitting clayton copula to linear scenario
    try:
        clayton_model.set_params(learning_rate=0.01, reg_lambda=0.1, max_depth=1)
        with suppress_stdout_stderr(): # suppress progress bar
            clayton1_xgb = clayton_model.fit(x_train_clayton1, y_train_clayton1, eval_set=[(x_val_clayton1, y_val_clayton1)])

        test_preds_clayton1 = clayton1_xgb.predict(x_test_clayton1)[:,0].reshape(-1,1)

        pdp_clayton1_x1 = partial_dependence_plot(
            clayton1_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau1,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_clayton1_x2 = partial_dependence_plot(
            clayton1_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau1,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_clayton1_x3 = partial_dependence_plot(
            clayton1_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau1,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_clayton1_x4 = partial_dependence_plot(
            clayton1_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau1,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        results["linear"]["clayton"]["pdp"]["x1"][i] = pdp_clayton1_x1
        results["linear"]["clayton"]["pdp"]["x2"][i] = pdp_clayton1_x2
        results["linear"]["clayton"]["pdp"]["x3"][i] = pdp_clayton1_x3
        results["linear"]["clayton"]["pdp"]["x4"][i] = pdp_clayton1_x4

        results["linear"]["clayton"]["feature_importance"][i] = clayton1_xgb.feature_importances_.tolist()

        results["linear"]["clayton"]["nll_xgb"].append(clayton.clayton_copula_nll_metric(y_test_clayton1, test_preds_clayton1))
        results["linear"]["clayton"]["nll_constant"].append(clayton.clayton_copula_nll_metric(y_test_clayton1, clayton.get_initial_guess(y_test_clayton1)))
        results["linear"]["clayton"]["mse_xgb"].append(mse(true_tau1(x_test_clayton1).reshape(-1,1), clayton.clayton_predt2tau(test_preds_clayton1)))
        results["linear"]["clayton"]["mse_constant"].append(mse(true_tau1(x_test_clayton1), clayton.clayton_predt2tau(clayton.get_initial_guess(y_test_clayton1))))
    except:
        pass



    # Fitting clayton copula to non-linear scenario
    try:
        clayton_model.set_params(learning_rate=0.01, reg_lambda=0.0, max_depth=1)
        with suppress_stdout_stderr(): # suppress progress bar
            clayton2_xgb = clayton_model.fit(x_train_clayton2, y_train_clayton2, eval_set=[(x_val_clayton2, y_val_clayton2)])

        test_preds_clayton2 = clayton2_xgb.predict(x_test_clayton2)[:,0].reshape(-1,1)

        pdp_clayton2_x1 = partial_dependence_plot(
            clayton2_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau2,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_clayton2_x2 = partial_dependence_plot(
            clayton2_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau2,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_clayton2_x3 = partial_dependence_plot(
            clayton2_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau2,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_clayton2_x4 = partial_dependence_plot(
            clayton2_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau2,
            transform=clayton.clayton_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        results["non-linear"]["clayton"]["pdp"]["x1"][i] = pdp_clayton2_x1
        results["non-linear"]["clayton"]["pdp"]["x2"][i] = pdp_clayton2_x2
        results["non-linear"]["clayton"]["pdp"]["x3"][i] = pdp_clayton2_x3
        results["non-linear"]["clayton"]["pdp"]["x4"][i] = pdp_clayton2_x4

        results["non-linear"]["clayton"]["feature_importance"][i] = clayton2_xgb.feature_importances_.tolist()

        results["non-linear"]["clayton"]["nll_xgb"].append(clayton.clayton_copula_nll_metric(y_test_clayton2, test_preds_clayton2))
        results["non-linear"]["clayton"]["nll_constant"].append(clayton.clayton_copula_nll_metric(y_test_clayton2, clayton.get_initial_guess(y_test_clayton2)))
        results["non-linear"]["clayton"]["mse_xgb"].append(mse(true_tau2(x_test_clayton2).reshape(-1,1), clayton.clayton_predt2tau(test_preds_clayton2)))
        results["non-linear"]["clayton"]["mse_constant"].append(mse(true_tau2(x_test_clayton2), clayton.clayton_predt2tau(clayton.get_initial_guess(y_test_clayton2))))
    except:
        pass

    # Gumbel copula
    y_gumbel1 = clip_copula_samples(gumbel.sample_gumbel_copula(gumbel.gumbel_tau2par(tau1), seed=seed), eps=1e-7)

    #train, validation and test data split (70,10,20)
    x_main_gumbel1, x_test_gumbel1, y_main_gumbel1, y_test_gumbel1 = train_test_split(x, y_gumbel1, test_size=0.2, random_state=42)
    x_train_gumbel1, x_val_gumbel1, y_train_gumbel1, y_val_gumbel1 = train_test_split(x_main_gumbel1, y_main_gumbel1, test_size=0.125, random_state=42)

    y_gumbel2 = clip_copula_samples(gumbel.sample_gumbel_copula(gumbel.gumbel_tau2par(tau2), seed=seed), eps=1e-7)

    #train, validation and test data split (70,10,20)
    x_main_gumbel2, x_test_gumbel2, y_main_gumbel2, y_test_gumbel2 = train_test_split(x, y_gumbel2, test_size=0.2, random_state=42)
    x_train_gumbel2, x_val_gumbel2, y_train_gumbel2, y_val_gumbel2 = train_test_split(x_main_gumbel2, y_main_gumbel2, test_size=0.125, random_state=42)

    # Fitting gumbel copula to linear scenario
    try:
        gumbel_model.set_params(learning_rate=0.1, reg_lambda=0.2, max_depth=1)
        with suppress_stdout_stderr(): # suppress progress bar
            gumbel1_xgb = gumbel_model.fit(x_train_gumbel1, y_train_gumbel1, eval_set=[(x_val_gumbel1, y_val_gumbel1)])

        test_preds_gumbel1 = gumbel1_xgb.predict(x_test_gumbel1)[:,0].reshape(-1,1)

        pdp_gumbel1_x1 = partial_dependence_plot(
            gumbel1_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau1,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_gumbel1_x2 = partial_dependence_plot(
            gumbel1_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau1,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_gumbel1_x3 = partial_dependence_plot(
            gumbel1_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau1,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_gumbel1_x4 = partial_dependence_plot(
            gumbel1_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau1,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        results["linear"]["gumbel"]["pdp"]["x1"][i] = pdp_gumbel1_x1
        results["linear"]["gumbel"]["pdp"]["x2"][i] = pdp_gumbel1_x2
        results["linear"]["gumbel"]["pdp"]["x3"][i] = pdp_gumbel1_x3
        results["linear"]["gumbel"]["pdp"]["x4"][i] = pdp_gumbel1_x4

        results["linear"]["gumbel"]["feature_importance"][i] = gumbel1_xgb.feature_importances_.tolist()

        results["linear"]["gumbel"]["nll_xgb"].append(gumbel.gumbel_copula_nll_metric(y_test_gumbel1, test_preds_gumbel1))
        results["linear"]["gumbel"]["nll_constant"].append(gumbel.gumbel_copula_nll_metric(y_test_gumbel1, gumbel.get_initial_guess(y_test_gumbel1)))
        results["linear"]["gumbel"]["mse_xgb"].append(mse(true_tau1(x_test_gumbel1).reshape(-1,1), gumbel.gumbel_predt2tau(test_preds_gumbel1)))
        results["linear"]["gumbel"]["mse_constant"].append(mse(true_tau1(x_test_gumbel1), gumbel.gumbel_predt2tau(gumbel.get_initial_guess(y_test_gumbel1))))
    except:
        pass


    # Fitting gumbel copula to non-linear scenario
    try:
        gumbel_model.set_params(learning_rate=0.01, reg_lambda=0.2, max_depth=2)
        with suppress_stdout_stderr(): # suppress progress bar
            gumbel2_xgb = gumbel_model.fit(x_train_gumbel2, y_train_gumbel2, eval_set=[(x_val_gumbel2, y_val_gumbel2)])

        test_preds_gumbel2 = gumbel2_xgb.predict(x_test_gumbel2)[:,0].reshape(-1,1)

        pdp_gumbel2_x1 = partial_dependence_plot(
            gumbel2_xgb,
            x_pdp,
            feature_index=0,
            true_theta_fn=true_tau2,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_gumbel2_x2 = partial_dependence_plot(
            gumbel2_xgb,
            x_pdp,
            feature_index=1,
            true_theta_fn=true_tau2,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_gumbel2_x3 = partial_dependence_plot(
            gumbel2_xgb,
            x_pdp,
            feature_index=2,
            true_theta_fn=true_tau2,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        pdp_gumbel2_x4 = partial_dependence_plot(
            gumbel2_xgb,
            x_pdp,
            feature_index=3,
            true_theta_fn=true_tau2,
            transform=gumbel.gumbel_predt2tau,
            grid = np.linspace(0,1,grid_resolution)
        )

        results["non-linear"]["gumbel"]["pdp"]["x1"][i] = pdp_gumbel2_x1
        results["non-linear"]["gumbel"]["pdp"]["x2"][i] = pdp_gumbel2_x2
        results["non-linear"]["gumbel"]["pdp"]["x3"][i] = pdp_gumbel2_x3
        results["non-linear"]["gumbel"]["pdp"]["x4"][i] = pdp_gumbel2_x4

        results["non-linear"]["gumbel"]["feature_importance"][i] = gumbel2_xgb.feature_importances_.tolist()

        results["non-linear"]["gumbel"]["nll_xgb"].append(gumbel.gumbel_copula_nll_metric(y_test_gumbel2, test_preds_gumbel2))
        results["non-linear"]["gumbel"]["nll_constant"].append(gumbel.gumbel_copula_nll_metric(y_test_gumbel2, gumbel.get_initial_guess(y_test_gumbel2)))
        results["non-linear"]["gumbel"]["mse_xgb"].append(mse(true_tau2(x_test_gumbel2).reshape(-1,1), gumbel.gumbel_predt2tau(test_preds_gumbel2)))
        results["non-linear"]["gumbel"]["mse_constant"].append(mse(true_tau2(x_test_gumbel2), gumbel.gumbel_predt2tau(gumbel.get_initial_guess(y_test_gumbel2))))
    except:
        pass

# write to json file
with open('phase1_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)