import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats
import scipy.special as special

class LognormalLikelihood(loss.Loss):
    """
    Probabilistic prediction / Parametric inference using copula-based multivariate negative log-likelihood loss functions.
    """
    required_pars = [] # might expand subclass later to accept copula and marginal families as parameters
    
    """"
    :param lambda_weights:  quadratic penalization parameter for the leaves weights
    :param lambda_leaves: quadratic penalization parameter for the number of leaves
    :param loss_kwargs: additional parameters needed for a specific loss type
    """
    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.n_dims = 2

    def set_dimension(self, n_dims):
        pass

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        shape, _, scale = stats.lognorm.fit(y, floc = 0)
        #eta2_hat = np.log(1/(2*shape**2))
        omega2_hat = np.log(1/(2*shape**2))
        eta1_hat = np.log(scale)/(shape**2)
        #initial_parameter_estimates = np.array([[eta1_hat, eta2_hat]])
        #print(initial_parameter_estimates)
        return np.array([[eta1_hat, omega2_hat]]) #initial_parameter_estimates
    
    #def lognormal_lpdf_grad_eta1(self, y, eta1, eta2):
        #return np.log(y) - eta1/(2*eta2)
    
    #def lognormal_lpdf_grad_eta2(self, y, eta1, eta2):
        #return -(np.log(y)**2) + (eta1/(2*eta2))**2 + 1/(2*eta2)
    
    #def lognormal_lpdf_hess_eta1(self, eta2):
        #return -1/(2*eta2)
    
    #def lognormal_lpdf_hess_eta2(self, eta1, eta2):
        #return -(eta1**2 + eta2)/(eta2**3)

    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        """
        :param y: observation matrix (n_obs, n_t) -> np.ndarray
        :param y_hat: current (transformed) parameter estimates matrix (n_obs, n_t) -> np.ndarray (e.g. y_hat = log(lambda) for exponential distribution assuming log link)
        :param iteration: current number of iteration, needed to indicate when warm-up phase ends
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix
        """
        eta1 = y_hat[:, 0].reshape(-1,1)
        omega2 = y_hat[:, 1].reshape(-1,1)
        #eta2 = np.exp(y_hat[:, 1]).reshape(-1,1)

        eta1_grad = eta1/(2*np.exp(omega2)) - np.log(y) #-self.lognormal_lpdf_grad_eta1(y, eta1, eta2)
        omega2_grad = -(eta1**2)/(4*np.exp(omega2)) - 1/2 + (np.log(y)**2)*np.exp(omega2)

        #eta2_grad = -self.lognormal_lpdf_grad_eta2(y, eta1, eta2)*eta2

        #if iteration == 0:
            #eta1_hess = -self.lognormal_lpdf_hess_eta1(eta2)*np.ones_like(eta1_grad)
            #eta2_hess = -(self.lognormal_lpdf_hess_eta2(eta1, eta2)*(eta2**2) - eta2_grad)*np.ones_like(eta2_grad)
        #else:
           #eta1_hess = -self.lognormal_lpdf_hess_eta1(eta2)
           #eta2_hess = -self.lognormal_lpdf_hess_eta2(eta1, eta2)*(eta2**2) + eta2_grad

        eta1_hess = 1/np.exp(omega2)
        omega2_hess = (3*(eta1**2))/(4*np.exp(omega2)) + 1/2 + (np.log(y)**2)*np.exp(omega2)

        # combine gradient components into a vector
        grad = np.hstack((eta1_grad, omega2_grad))

        # combine hessian components into a vector
        hess_diags = np.hstack((eta1_hess*np.ones_like(omega2_hess), omega2_hess))
        
        return grad, hess_diags
    
    def H_inv(self, H):
        """
        Computes the inverse of the Hessian, given the Hessian's diagonal of the current leave.

        :param H: current leaf Hessian's diagonal (n_t, 1)
        """
        return np.diagflat(1 / (H + self.lambda_weights))

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
        eta1 = y_hat[:, 0].reshape(-1,1)
        omega2 = y_hat[:, 1].reshape(-1,1)
        
        nll = -np.sum(stats.lognorm.logpdf(x = y, scale = np.exp(eta1/(2*np.exp(omega2))), s = 1/(2*np.exp(omega2))))
        return nll