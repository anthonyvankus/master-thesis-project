import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats

class CopulaLikelihood(loss.Loss):
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
    
    def set_dimension(self, n_dims):
        self.n_dims = 3 #note: for our simple bicopula example with exponential marginals we will hardcode the number of dimensions = number of parameters = 3

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        return np.hstack((np.log(1/np.mean(y, axis = 0)*np.ones_like(y)), self.fgm_par_link(np.zeros((y.shape[0], 1)))))

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
    
    def fgm_copula_lpdf(self, u, par):
        """
        :param u: bivariate pseudo-observation vector (n_obs, 2) where u = CDF(obs)
        :param par: current dependence parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: log pdf of a bivariate FGM copula dist
        """
        u0 = u[:,0].reshape(-1,1)
        u1 = u[:,1].reshape(-1,1)
        return np.sum(np.log(1 + par*(1-2*u0)*(1-2*u1)))
    
    def fgm_copula_lpdf_grad(self, u, par):
        """
        :param u: bivariate pseudo-observation vector (n_obs, 2) where u = CDF(obs)
        :param par: current dependence parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: first derivative of log pdf of a bivariate FGM copula dist w.r.t its parameter
        """
        u0 = u[:,0].reshape(-1,1)
        u1 = u[:,1].reshape(-1,1)
        prod = (1-2*u0)*(1-2*u1) # (n_obs, 1)
        return prod/(1 + par*prod) # (n_obs, 1)

    def fgm_copula_lpdf_hess(self, u, par):
        """
        :param u: bivariate pseudo-observation vector (n_obs, 2) where u = CDF(obs)
        :param par: current parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: second derivative of log pdf of a bivariate FGM copula dist w.r.t its parameter
        """
        return -np.square(self.fgm_copula_lpdf_grad(u, par))
    
    def fgm_par_link(self, par):
        """
        :param par: current parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: link function of FGM copula parameter
        """
        return np.tan((np.pi/2)*par)
    
    def fgm_par_inv_link(self, tpar):
        """
        :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: inverse link function of transformed FGM copula parameter
        """
        return (2/np.pi)*np.arctan(tpar)
    
    def fgm_par_inv_link_grad(self, tpar):
        """
        :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: gradient of inverse link function w.r.t transformed FGM copula parameter
        """
        return (2/np.pi)*(1/(np.square(tpar)+1))
    
    def fgm_par_inv_link_hess(self, tpar):
        """
        :param tpar: current (transformed) parameter estimates (n_obs, n_param) (e.g. for a bivariate FGM copula, n_param = 1)

        :return: second derivative of inverse link function w.r.t. transformed FGM copula parameter
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
        y0 = y[:, 0].reshape(-1,1)
        y1 = y[:, 1].reshape(-1,1)
        lambda0 = y_hat[:, 0].reshape(-1,1)
        lambda1 = y_hat[:, 1].reshape(-1,1)
        rho = y_hat[:, -1].reshape(-1, 1)

        # get gradient for each marginal dist parameters
        m0_grad = self.exp_lpdf_grad(y0, np.exp(lambda0))*np.exp(lambda0) # log link is used for exponential parameters
        m1_grad = self.exp_lpdf_grad(y1, np.exp(lambda1))*np.exp(lambda1)

        # compute pseudo-observations for FGM copula given estimated marginal dist parameters (subject to change: e.g. rank-based pseudo-observations could be used instead)
        u1 = stats.expon.cdf(y0, loc = 0, scale = 1/np.exp(lambda0))
        u2 = stats.expon.cdf(y1, loc = 0, scale = 1/np.exp(lambda1))
        u = np.hstack((u1, u2))
        
        # get gradient for copula parameter
        cop_grad = self.fgm_copula_lpdf_grad(u, self.fgm_par_inv_link(rho))*self.fgm_par_inv_link_grad(rho)
        
        # combine grad components into a vector
        grad = np.hstack((m0_grad, m1_grad, cop_grad))

        # get second derivative for each marginal dist parameters
        m0_hess = self.exp_lpdf_hess(np.exp(lambda0))*np.square(np.exp(lambda0)) + m0_grad #special case: m0_grad = g^IFM * d^2 u^-1 / d lambda0^2
        m1_hess = self.exp_lpdf_hess(np.exp(lambda1))*np.square(np.exp(lambda1)) + m1_grad #special case: m1_grad = g^IFM * d^2 u^-1 / d lambda1^2
        cop_hess = self.fgm_copula_lpdf_hess(u, self.fgm_par_inv_link(rho))*np.square(self.fgm_par_inv_link_grad(rho)) + self.fgm_copula_lpdf_grad(u, self.fgm_par_inv_link(rho))*self.fgm_par_inv_link_hess(rho)

        # combine hessian components into a vector representing ONLY the diagonal of the IFM hessian
        hess_diags = np.hstack((m0_hess, m1_hess, cop_hess))

        return grad, hess_diags
    
    def H_inv_last_row_effect(self, G, H):
        """
        Evaluate the effect of the last row from the IFM-approximated inverse Hessian

        :param G: mean gradient for the current leaf. (n_t) square of mean gradient can also be passed.
        :param H: mean Hessian for the current leaf. (n_t)

        :return: correction term for optimal response of copula dependence parameter (scalar)
        """
        const = G[-1]/(H[-1] + self.lambda_weights)
        return const * G[:-1] @ (1/(H[:-1] + self.lambda_weights))
    
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
        :param H: sum of Hessians diags in the current leaf

        :return: optimal loss, scalar
        """
        H_inv = self.H_inv(H)
        optimal_loss = - np.sum(H_inv @ G2)
        optimal_loss += - self.H_inv_last_row_effect(G2, H) # adjusting for the effects of the last row
        return optimal_loss
    
    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: mean gradient for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        H_inv = self.H_inv(H)
        optimal_response = H_inv @ G
        optimal_response[-1] += self.H_inv_last_row_effect(G, H) # adjusting for the effects of the last row
        return optimal_response

    def tree_loss(self, y, y_hat):
        """
        Compute the tree loss (without penalizations). The loss is given by the negative log-likelihood.

        :param y: observations of the target on the training set
        :param y_hat: current parameter estimation of the MBT

        :return: tree loss
        """
        u1 = stats.expon.cdf(y[:, 0].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 0].reshape(-1, 1)))
        u2 = stats.expon.cdf(y[:, 1].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 1].reshape(-1, 1)))
        u = np.hstack((u1, u2))
        return -(self.fgm_copula_lpdf(u, self.fgm_par_inv_link(y_hat[:, -1].reshape(-1, 1))) + np.sum(stats.expon.logpdf(y[:, 0].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 0].reshape(-1, 1)))) + np.sum(stats.expon.logpdf(y[:, 1].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 1].reshape(-1, 1)))))

