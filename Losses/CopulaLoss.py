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
        self.n_dims = 3
        #self.cov_mat = np.eye(self.n_dims)

    def set_dimension(self, n_dims):
        pass
        #self.n_dims = 3 #note: for our simple bicopula example with exponential marginals we will hardcode the number of dimensions = number of parameters = 3

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        return np.hstack((np.log(1/np.mean(y, axis = 0)*np.ones_like(y)), np.zeros((y.shape[0], 1)))) # np.log(1/np.mean(y, axis = 0)*np.ones_like(y))
    
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

    def exp_cdf_grad(self, y, par):
        """
        :param y: univariate observation vector
        :param par: current parameter estimates (n_obs, 1)

        :return: derivative of the exponential cdf wrt its parameter
        """
        return y*np.exp(-par*y)
    
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
    
    def fgm_lpdf_mixed_hess(self, u, v, par):
        """
        Calculates the derivative of the FGM lpdf wrt to u and rho

        :param u: first marginal pseudo-observations
        :param v: second marginal pseudo-observations
        :param par: current parameter estimates

        :return: mixed derivative of FGM lpdf (dl_c/dudp)
        """
        prod = (1-2*u)*(1-2*v) # (n_obs, 1)
        return -2*(1-2*v)/np.square(1 + par*prod) # (n_obs, 1)

    def hessian_sum_split(self, H):
        """
        Splits flattened Hessian into diagonal and last row components. A copy of the gradient is also retrieved

        :param H: flattened hessian vector

        :return: diag, last row and grad
        """
        diag = H[:self.n_dims]
        #diag = H[:self.n_dims**2]
        #grad = H[self.n_dims**2:-1]
        last_row = H[self.n_dims:2*self.n_dims-1]
        grad = H[2*self.n_dims-1:]
        #n = H[-1]
        return diag, last_row, grad

    def BHHH_outer_product_flat(self, grad):
        outer_products = np.einsum('ni, nj->nij', grad, grad)
        return outer_products.reshape((outer_products.shape[0], -1))

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
        m0_grad = -self.exp_lpdf_grad(y0, np.exp(lambda0))*np.exp(lambda0) # log link is used for exponential parameters
        m1_grad = -self.exp_lpdf_grad(y1, np.exp(lambda1))*np.exp(lambda1)

        #marg_grad = np.hstack((m0_grad, m1_grad))

        # compute pseudo-observations for FGM copula given marginal CDF from estimated dist. parameters (subject to change: e.g. rank-based pseudo-observations could be used instead)
        #u1 = stats.expon.cdf(y0, loc = 0, scale = 1/np.exp(lambda0))
        #u2 = stats.expon.cdf(y1, loc = 0, scale = 1/np.exp(lambda1))
        #u = np.hstack((u1, u2))

        # rank-based method of computing pseudo-observations
        u = self.rank_pseudo_obs(y)

        # get gradient for copula parameter
        cop_grad = -self.fgm_copula_lpdf_grad(u, self.fgm_par_inv_link(rho))*self.fgm_par_inv_link_grad(rho)
        
        # combine grad components into a vector
        grad = np.hstack((m0_grad, m1_grad, cop_grad))
        #grad = np.hstack((m0_grad, m1_grad))

        #TEST
        #cov_mat = np.cov(m = grad, rowvar=False)
        #self.cov_mat = cov_mat
        #outer_products_flat = self.BHHH_outer_product_flat(grad)
        #outer_product_sum = np.sum(outer_products_flat, axis = 0).reshape(self.n_dims, self.n_dims)
        #print(outer_product_sum)
        #print(cov_mat)

        # get second derivative for each marginal dist parameters
        m0_hess = -self.exp_lpdf_hess(np.exp(lambda0))*np.square(np.exp(lambda0)) + m0_grad #special case: m0_grad = g^IFM * d^2 u^-1 / d lambda0^2
        m1_hess = -self.exp_lpdf_hess(np.exp(lambda1))*np.square(np.exp(lambda1)) + m1_grad #special case: m1_grad = g^IFM * d^2 u^-1 / d lambda1^2
        cop_hess = -self.fgm_copula_lpdf_hess(u, self.fgm_par_inv_link(rho))*np.square(self.fgm_par_inv_link_grad(rho)) + self.fgm_copula_lpdf_grad(u, self.fgm_par_inv_link(rho))*self.fgm_par_inv_link_hess(rho)

        # compute last row component
        # hess_last_row = cop_grad * marg_grad

        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)
        hess_last_row0 = -self.fgm_lpdf_mixed_hess(u0, u1, self.fgm_par_inv_link(rho))*self.fgm_par_inv_link_grad(rho)*np.exp(lambda0)*self.exp_cdf_grad(y[:, 0].reshape(-1,1), np.exp(lambda0))
        hess_last_row1 = -self.fgm_lpdf_mixed_hess(u1, u0, self.fgm_par_inv_link(rho))*self.fgm_par_inv_link_grad(rho)*np.exp(lambda1)*self.exp_cdf_grad(y[:, 1].reshape(-1,1), np.exp(lambda1))

        #print(hess_last_row0)
        #print(hess_last_row1)
        #print(self.fgm_lpdf_mixed_hess(u0, u1, self.fgm_par_inv_link(rho)).shape)
        #print(np.sum(np.hstack((m0_hess, m1_hess, cop_hess, hess_last_row0, hess_last_row1)), axis = 0))
        # combine hessian components into a vector representing the diagonal of the IFM hessian PLUS the last IFM hessian row PLUS a copy of the gradient
        #hess_diags = np.hstack((m0_hess, m1_hess, cop_hess, hess_last_row0, hess_last_row1, grad, np.ones_like(m0_hess)))
        #hess_diags = np.hstack((m0_hess, m1_hess, cop_hess))
        #print(np.sum(hess_diags, axis = 0))
        #hess_diags = np.hstack((outer_products_flat, grad, np.ones_like(m0_grad)))
        hess_diags = np.hstack((m0_hess, m1_hess, cop_hess, hess_last_row0, hess_last_row1, grad))

        #TEST
        #hess_diags = np.hstack((np.ones_like(grad), grad))
        #hess_diags = np.hstack((m0_hess, m1_hess))

        return grad, hess_diags
    
    def H_inv_last_row_effect_loss(self, G2, H):
        """
        Evaluate the effect of the last row from the IFM-approximated inverse Hessian for the loss

        :param G2: mean square gradient for the current leaf. (n_t)
        :param H: mean Hessian for the current leaf. (n_t)

        :return: correction term for optimal loss of copula dependence parameter (scalar)
        """
        #const = G2[-1]/(H[-1] + self.lambda_weights)
        #return const * G2[:-1] @ (1/(H[:-1] + self.lambda_weights))
        pass
    
    def H_inv_last_row_effect_response(self, G, H):
        """
        Evaluate the effect of the last row from the IFM-approximated inverse Hessian for the response

        :param G: negative mean gradient for the current leaf. (n_t)
        :param H: mean Hessian for the current leaf. (n_t)

        :return: correction term for optimal response of copula dependence parameter (scalar)
        """
        pass
        #inv_last_row = H/
        #const = 1/(H[-1] + self.lambda_weights)
        #return #const * np.square(G) @ (1/(H[:-1] + self.lambda_weights))

    
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
        D, E, G = self.hessian_sum_split(H)
        #M, grad, n = self.hessian_sum_split(H)
        #diag, grad = self.hessian_sum_split(H)
        #D, E, G, leaf_size = self.hessian_sum_split(H)
        A = np.diag(D + self.lambda_weights)
        #A[:-1, -1] += E
        A[-1, :-1] += E
        H_inv = np.linalg.inv(A)
        optimal_loss = -(G.T @ H_inv @ G)
        #optimal_loss = - np.sum(H_inv @ G2)
        #last_row_inv = grad[-1]*self.H_inv_last_row(D, E)
        #optimal_loss += - last_row_inv @ grad[: -1] # adjusting for the effects of the last row
        #leaf_size = diag[0]
        #optimal_loss = -(G.T @ self.cov_mat @ G)/leaf_size
        #print(grad)
        #print(self.cov_mat/leaf_size)
        #H_inv = self.H_inv(H)
        #G.reshape(-1, 1)
        
        #optimal_loss = -(G.T @ (H_inv @ G))
        #optimal_loss = -np.sum(H_inv @ G2)
        #outer_product_sum = M.reshape(self.n_dims, self.n_dims)
        #H_inv = np.linalg.inv(outer_product_sum)
        #optimal_loss = -(grad.T @ H_inv @ grad)
        return optimal_loss
    
    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: negative sum of gradients for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        D, E, _ = self.hessian_sum_split(H)
        #M, _, n = self.hessian_sum_split(H)
        #D, E, _, leaf_size = self.hessian_sum_split(H)
        A = np.diag(D + self.lambda_weights)
        #A[:-1, -1] += E
        A[-1, :-1] += E
        H_inv = np.linalg.inv(A)
        optimal_response = H_inv @ G
        #last_row_inv = self.H_inv_last_row(D, E)
        #optimal_response[-1] += last_row_inv @ G[: -1] # adjusting for the effects of the last row
        #leaf_size = diag[0]
        #optimal_response = (self.cov_mat @ G)/leaf_size
        #H_inv = self.H_inv(H)
        #optimal_response = H_inv @ G
        #outer_product_sum = M.reshape(self.n_dims, self.n_dims)
        #H_inv = np.linalg.inv(outer_product_sum)
        #optimal_response = (H_inv @ G)
        return optimal_response

    def tree_loss(self, y, y_hat):
        """
        Compute the tree loss (without penalizations). The loss is given by the negative log-likelihood.

        :param y: observations of the target on the training set
        :param y_hat: current parameter estimation of the MBT

        :return: tree loss
        """
        #u1 = stats.expon.cdf(y[:, 0].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 0].reshape(-1, 1)))
        #u2 = stats.expon.cdf(y[:, 1].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 1].reshape(-1, 1)))
        #u = np.hstack((u1, u2))

        u = self.rank_pseudo_obs(y)
        return -(self.fgm_copula_lpdf(u, self.fgm_par_inv_link(y_hat[:, -1].reshape(-1, 1))) + np.sum(stats.expon.logpdf(y[:, 0].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 0].reshape(-1, 1)))) + np.sum(stats.expon.logpdf(y[:, 1].reshape(-1, 1), loc = 0, scale = 1/np.exp(y_hat[:, 1].reshape(-1, 1)))))

