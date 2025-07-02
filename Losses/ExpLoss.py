import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats

class JointExpLikelihood(loss.Loss):
    """
    Probabilistic prediction / Parametric inference using copula-based multivariate negative log-likelihood loss functions.
    """
    required_pars = [] #empty for now, might expand subclass later to accept copula and marginal families as parameters
    
    """"
    :param lambda_weights:  quadratic penalization parameter for the leaves weights
    :param lambda_leaves: quadratic penalization parameter for the number of leaves
    :param loss_kwargs: additional parameters needed for a specific loss type
    """
    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        self.n_dims = 2

    def set_dimension(self, n_dims):
        pass

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        return np.log(1/np.mean(y, axis = 0)*np.ones_like(y))
    
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

    def exp_lpdf_grad(self, y, par):
        """
        :param y: univariate observation vector (n_obs, 1)
        :param par: current parameter estimates (n_obs, 1)

        :return: first derivative of log pdf of an exponential dist w.r.t its parameter
        """
        return 1/par - y

    def exp_lpdf_hess(self, par):
        """
        :param par: current parameter estimates (n_obs, 1)

        :return: second derivative of log pdf of an exponential dist w.r.t its parameter
        """
        return -1/np.square(par)
    
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
        y0 = y[:, 0].reshape(-1,1)
        y1 = y[:, 1].reshape(-1,1)
        lambda0 = y_hat[:, 0].reshape(-1,1)
        lambda1 = y_hat[:, 1].reshape(-1,1)

        # get gradient for each marginal dist parameters
        m0_grad = -self.exp_lpdf_grad(y0, np.exp(lambda0))*np.exp(lambda0) # log link is used for exponential parameters
        m1_grad = -self.exp_lpdf_grad(y1, np.exp(lambda1))*np.exp(lambda1)

        # combine grad components into a vector
        grad = np.hstack((m0_grad, m1_grad))
       
        # get second derivative for each marginal dist parameters
        m0_hess = -self.exp_lpdf_hess(np.exp(lambda0))*np.square(np.exp(lambda0)) + m0_grad #special case: m0_grad = g^IFM * d^2 u^-1 / d lambda0^2
        m1_hess = -self.exp_lpdf_hess(np.exp(lambda1))*np.square(np.exp(lambda1)) + m1_grad #special case: m1_grad = g^IFM * d^2 u^-1 / d lambda1^2

        #combine hess component into a vector
        hess_diags = np.hstack((m0_hess, m1_hess))

        return grad, hess_diags
    
    def H_inv(self, H):
        """
        Computes the inverse of the Hessian, given the Hessian's diagonal of the current leave.

        :param H: current leaf Hessian's diagonal (n_t, 1)
        """
        return np.diag(1 / (H + self.lambda_weights))

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
        return -np.sum(np.log(np.exp(y_hat)) - np.exp(y_hat)*y)