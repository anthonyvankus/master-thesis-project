import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats
import scipy.special as special
#import vinecopulas as vine

class BiGumbelCopLikelihood(loss.Loss):
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

    def gumbel_tau2par(self, tau):
        """
        Return Gumbel parameters from Kendall's tau values
        """
        return 1/(1-tau)

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the dependence parameters to Kendall's tau estimates for the entire dataset of pseudo-observations

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        u0 = y[:, 0].reshape(-1,1)
        u1 = y[:, 1].reshape(-1,1)
        tau, _ = stats.kendalltau(u0, u1)
        #theta = self.gumbel_tau2par(tau)
        #print("yes1")

        return special.logit(tau)*np.ones((y.shape[0], 1)) #np.log(theta - 1)*np.ones((y.shape[0], 1)) # log link as reparametrization

    def gumbel_lpdf(self, u, theta):
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)
        t = (-np.log(u0))**theta
        s = (-np.log(u1))**theta
        a = (t + s)**(1/theta)

        return -a - np.log(u0*u1) + (-2+2/theta)*np.log(t+s) + (theta - 1)*(np.log(np.log(u0)*np.log(u1))) + np.log(1+(theta-1)/a)
    
    def gumbel_lpdf_grad(self, u, theta):
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
    
    def gumbel_lpdf_hess(self, u, theta):
        """
        Gumbel copula lpdf hessian w.r.t dependence parameter theta

        :param u: bivariate pseudo-observation vector (n_obs, 2)
        :param theta: current dependence parameter estimates (n_obs, 1)
        
        :return: Gumbel copula lpdf hessian
        """
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)
        t = (-np.log(u0))**theta
        s = (-np.log(u1))**theta
        a = (t + s)**(1/theta)
        b = np.log(t+s)/np.square(theta)
        d = (t*np.log(-np.log(u0)) + s*np.log(-np.log(u1)))/(t+s)

        dtd0 = t*np.log(-np.log(u0))
        dsd0 = s*np.log(-np.log(u1))
        dad0 = ((-a*np.log(t+s))/np.square(theta))*(dtd0 + dsd0)
        dbd0 = (dtd0 + dsd0)/(t+s) - (2/theta)*np.log(t+s)
        ddd0 = (t*(np.log(-np.log(u0)))**2 + s*(np.log(-np.log(u1)))**2)/(t+s) - ((dtd0 + dsd0)/(t+s))**2

        dAd0 = dad0*(b - d/theta) + a*(dbd0 - ddd0/theta + d/(theta**2))
        dBd0 = -2*(dbd0 + ddd0 -2*(ddd0/theta - d/(theta**2)))
        dCd0 = ((theta - 1)*dad0)/((a + theta - 1)**2)
        dDd0 = (-1/np.square(a))*dad0 + (1/a)*((b - d/theta) + (theta - 1)*(dbd0 - ddd0/theta + d/(theta**2))) + (-1/np.square(a))*dad0*(theta - 1)*(b - d/theta)

        C = a/(a + theta - 1)
        D = 1/a + ((theta - 1)/a)*(b - d/theta)

        return dAd0 + dBd0 + (C*dDd0 - dCd0*D)/np.square(C)

    def gumbel_fisher_information(self, theta):
        K0 = 5/6 - (np.pi**2)/18
        return (1/theta**4)*((theta**2)*(-2/3 + (np.pi**2)/9) - theta + 2*K0/theta + (theta**3 + theta**2 + (K0-1)*theta -2*K0 + K0/theta)*special.exp1(theta-1)*np.exp(theta-1))
    
    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        """
        :param y: observation matrix (n_obs, n_t) -> np.ndarray
        :param y_hat: current (transformed) parameter estimates matrix (n_obs, n_t) -> np.ndarray (e.g. y_hat = log(lambda) for exponential distribution assuming log link)
        :param iteration: current number of iteration, generally not needed
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: IFM-approximated grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix
        """
        # define and reshape observation and parameter matrices 
        u0 = y[:, 0].reshape(-1,1)
        u1 = y[:, 1].reshape(-1,1)
        u = np.hstack((u0, u1)) # we assume the estimated pseudo-observations are passed on

        # define current dependence parameter
        tau = special.expit(y_hat)
        theta = self.gumbel_tau2par(tau)
        #theta = (np.exp(y_hat[:, 0]) + 1).reshape(-1,1)

        # get copula parameter gradient
        #cop_grad = -self.gumbel_lpdf_grad(u, theta)*theta
        cop_grad = self.gumbel_lpdf_grad(u,theta)*(1/(1-tau)**2)
        #print(np.isnan(cop_grad))

        # get copula parameter hessian
        #cop_hess = -self.gumbel_lpdf_hess(u, theta)*np.square(theta) + cop_grad
        #print(np.max(cop_hess))
        #print(np.min(cop_hess))

        # combine grad components into a vector
        grad = -cop_grad*(tau*(1-tau))

        #combine hess component into a vector
        hess_diags = grad**2 #np.ones((y.shape[0], 1)) #cop_hess #self.gumbel_fisher_information(theta)*np.square(theta)

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
        #theta = (np.exp(y_hat[:, 0])+1).reshape(-1,1)
        tau = special.expit(y_hat)
        theta = self.gumbel_tau2par(tau)

        return -np.sum(self.gumbel_lpdf(u, theta))