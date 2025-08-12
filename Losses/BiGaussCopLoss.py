import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats
import scipy.special as special

class BiGaussCopLikelihood(loss.Loss):
    """
    Probabilistic prediction / Parametric inference using copula-based multivariate negative log-likelihood loss functions.
    """
    required_pars = [] #empty
    
    """"
    :param lambda_weights:  quadratic penalization parameter for the leaves weights
    :param lambda_leaves: quadratic penalization parameter for the number of leaves
    :param loss_kwargs: additional parameters needed for a specific loss type
    """
    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        self.n_dims = 1

    def set_dimension(self, n_dims):
        pass

    def gauss_tau2par(self, tau):
         """
        Return Gaussian parameters from Kendall's tau values
        """
         return np.sin((np.pi/2)*tau)
    
    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        u0 = y[:, 0].reshape(-1,1)
        u1 = y[:, 1].reshape(-1,1)
        tau, _ = stats.kendalltau(u0, u1)
        #theta = self.gauss_tau2par(tau)
        #print("yes1")

        return special.logit(tau)*np.ones((y.shape[0], 1)) #self.cop_par_link(theta)*np.ones((y.shape[0], 1))
    
    def rank_pseudo_obs(self, y):
        """
        Return rank-based pseudo-observations from initial target matrix

        :param y: observation matrix (n_obs, 2)

        :return: rank-based pseudo-observations matrix (n_obs, 2)
        """
        n_obs = y.shape[0]
        u1 = stats.rankdata(y[:, 0])/(n_obs + 1)
        u2 = stats.rankdata(y[:, 1])/(n_obs + 1)
        return np.hstack((u1.reshape(-1,1), u2.reshape(-1,1)))

    def gaussian_lpdf(self, u, rho):
        a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
        b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
        c = a*b
        d = a**2 + b**2
        return np.sum(-np.log(np.sqrt(1-rho**2)) - ((rho**2)*d - 2*rho*c)/(2*(1-rho**2)))

    def gaussian_lpdf_grad(self, u, rho):
        """
        Gaussian copula gradient w.r.t dependence parameter rho

        :param u: bivariate pseudo-observation vector (n_obs, 2)
        :param rho: current dependence parameter estimates (n_obs, 1)
        
        :return: Gaussian copula lpdf gradient
        """
        a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
        b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
        c = a*b
        d = a**2 + b**2
        return (rho/(1-rho**2) + ((1+rho**2)*c - d*rho)/((1-rho**2)**2))
    
    def gaussian_lpdf_hess(self, u, rho):
        """
        Gaussian copula lpdf hessian w.r.t dependence parameter rho

        :param u: bivariate pseudo-observation vector (n_obs, 2)
        :param rho: current dependence parameter estimates (n_obs, 1)
        
        :return: Gaussian copula lpdf hessian
        """
        a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
        b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
        c = a*b
        d = a**2 + b**2
        return ((1+rho**2) + 2*c*rho - d)/((1-rho**2)**2) + (4*rho*(c*(1+rho**2) - d*rho))/((1-rho**2)**3) 
    
    def cop_par_link(self, par):
        """
        :param par: current parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

        :return: link function of copula parameter
        """
        return np.tan((np.pi/2)*par)
    
    def cop_par_inv_link(self, tpar):
        """
        :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

        :return: inverse link function of transformed copula parameter
        """
        return (2/np.pi)*np.arctan(tpar)
    
    def cop_par_inv_link_grad(self, tpar):
        """
        :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

        :return: gradient of inverse link function w.r.t transformed copula parameter
        """
        return (2/np.pi)*(1/(np.square(tpar)+1))
    
    def cop_par_inv_link_hess(self, tpar):
        """
        :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate copula, n_param = 1)

        :return: second derivative of inverse link function w.r.t. transformed copula parameter
        """
        return ((-4*tpar)/(np.pi*(np.square(tpar)+1)**2))
    
    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        """
        :param y: observation matrix (n_obs, n_t) -> np.ndarray
        :param y_hat: current (transformed) parameter estimates matrix (n_obs, n_t) -> np.ndarray (e.g. y_hat = log(lambda) for exponential distribution assuming log link)
        :param iteration: current number of iteration, generally not needed
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: IFM-approximated grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix

        note: for now we assume Exponential marginals and a bivariate FGM copula with one parameter
        """
        # define and reshape observation and parameter matrices 
        u0 = y[:, 0].reshape(-1,1)
        u1 = y[:, 1].reshape(-1,1)
        u = np.hstack((u0, u1)) # we assume the estimated pseudo-observations are passed on

        # define current dependence parameter
        #rho = self.cop_par_inv_link(y_hat[:, 0]).reshape(-1,1)
        tau = special.expit(y_hat)
        rho = self.gauss_tau2par(tau)

        # get copula parameter gradient
        cop_grad = self.gaussian_lpdf_grad(u, rho)*(np.pi/2)*np.cos((np.pi/2)*tau)#-self.gaussian_lpdf_grad(u, rho)*self.cop_par_inv_link_grad(self.cop_par_link(rho))

        # get copula parameter hessian
        #cop_hess = -(self.gaussian_lpdf_hess(u, rho)*(self.cop_par_inv_link_grad(self.cop_par_link(rho)))**2 + self.gaussian_lpdf_grad(u, rho)*self.cop_par_inv_link_hess(self.cop_par_link(rho))) #cop_grad**2 #np.ones((y.shape[0], 1))
        #print(np.max(cop_hess))
        #print(np.min(cop_hess))
        # combine grad components into a vector
        grad = -cop_grad*(tau*(1-tau))

        #combine hess component into a vector
        hess_diags = grad**2 #cop_hess #cop_grad**2 #-(np.ones((y.shape[0], 1))*(self.cop_par_inv_link_grad(self.cop_par_link(rho)))**2 + self.gaussian_lpdf_grad(u, rho)*self.cop_par_inv_link_hess(self.cop_par_link(rho)))

        return grad, hess_diags
    
    def H_inv(self, H):
        """
        Computes the inverse of the Hessian, given the Hessian's diagonal of the current leave.

        :param H: current leaf Hessian's diagonal (n_t, 1)
        """
        # (Dauphin et al., 2014) saddle-free Hessian computation requires we take the absolute value of the hessian
        return np.diag(1 / (np.abs(H) + self.lambda_weights))

    def eval_optimal_loss(self, G2, H):
        """
        Evaluate the optimal loss (using response obtained minimizing the second order loss approximation).

        :param G2: squared sum of gradients in the current leaf
        :param H: sum of Hessians diags 

        :return: optimal loss, scalar
        """
        H_inv = self.H_inv(H)
        optimal_loss = - np.sum(H_inv @ G2)
        return optimal_loss
    
    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: negative sum of gradients for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        H_inv = self.H_inv(H)
        optimal_response = H_inv @ G
        return optimal_response

    def tree_loss(self, y, y_hat):
        """
        Compute the tree loss (without penalizations). The loss is given by the negative log-likelihood.

        :param y: observations of the target on the training set
        :param y_hat: current parameter estimation of the MBT

        :return: tree loss
        """
        # define and reshape observation and parameter matrices 
        u0 = y[:, 0].reshape(-1,1)
        u1 = y[:, 1].reshape(-1,1)
        u = np.hstack((u0, u1)) # we assume pseudo-observations are passed on

        # define current dependence parameter
        #rho = self.cop_par_inv_link(y_hat[:, 0]).reshape(-1,1)
        tau = special.expit(y_hat)
        rho = self.gauss_tau2par(tau)

        return -np.sum(self.gaussian_lpdf(u, rho))