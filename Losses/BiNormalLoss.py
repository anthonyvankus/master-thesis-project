import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats

class BiNormalLikelihood(loss.Loss):
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
        self.n_dims = 5

    def set_dimension(self, n_dims):
        pass

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess i.e. (mu0, sigma0, mu1, sigma1, rho)
        """
        return np.hstack((np.mean(y[:,0], axis = 0)*np.ones((y.shape[0], 1)), np.log(np.std(y[:,0]))*np.ones((y.shape[0], 1)), np.mean(y[:,1], axis = 0)*np.ones((y.shape[0], 1)), np.log(np.std(y[:,1]))*np.ones((y.shape[0], 1)), np.zeros((y.shape[0], 1))))
    
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
    
    def norm_lpdf(self, y, mu, sigma):
        """
        :param y: univariate observation vector (n_obs, 1)
        :param mu: current location parameter estimates (n_obs, 1)
        :param sigma: current scale parameter estimates (n_obs, 1)

        :return: log pdf of a normal dist w.r.t location parameter
        """
        return np.sum(-np.log(sigma) - 0.5*(((y-mu)/sigma)**2))

    def norm_lpdf_mu_grad(self, y, mu, sigma):
        """
        :param y: univariate observation vector (n_obs, 1)
        :param mu: current location parameter estimates (n_obs, 1)
        :param sigma: current scale parameter estimates (n_obs, 1)

        :return: first derivative of log pdf of a normal dist w.r.t location parameter
        """
        return (y-mu)/np.square(sigma)

    def norm_lpdf_sigma_grad(self, y, mu, sigma):
        """
        :param y: univariate observation vector (n_obs, 1)
        :param mu: current location parameter estimates (n_obs, 1)
        :param sigma: current scale parameter estimates (n_obs, 1)

        :return: first derivative of log pdf of a normal dist w.r.t scale parameter
        """
        return -1/sigma + ((y-mu)**2)/(sigma**3)
    
    def norm_lpdf_mixed_hess(self, y, mu, sigma):
        return -2*((y-mu)/(sigma**3))
                   
    def norm_lpdf_mu_hess(self, sigma):
        """
        :param sigma: current scale parameter estimates (n_obs, 1)

        :return: second derivative of log pdf of a normal dist w.r.t location parameter
        """
        return -1/(sigma**2)

    def norm_lpdf_sigma_hess(self, y, mu, sigma):
        """
        :param y: univariate observation vector (n_obs, 1)
        :param mu: current location parameter estimates (n_obs, 1)
        :param sigma: current scale parameter estimates (n_obs, 1)

        :return: second derivative of log pdf of a normal dist w.r.t scale parameter
        """
        return 1/(sigma**2) - 3*(((y - mu)**2)/(sigma**4))
    
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
    
    def gaussian_lpdf_mixed_hess(y, u, mu, sigma):
        a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
        b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
        pass
    
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
    
    def hessian_sum_split(self, H):
        """
        Splits flattened Hessian into diagonal and last row components. A copy of the gradient is also retrieved

        :param H: flattened hessian vector

        :return: diag, last row and grad
        """

        diag = H[:self.n_dims]
        off_diag = H[self.n_dims:self.n_dims+2]
        grad = H[self.n_dims+2:]
        #diag = H[:self.n_dims**2]
        #grad = H[self.n_dims**2:-1]
        #last_row = H[self.n_dims:2*self.n_dims-1]
        #grad = H[2*self.n_dims-1:]
        #n = H[-1]
        return diag, off_diag, grad #diag, last_row, grad

    def BHHH_outer_product_flat(self, grad):
        """
        Calculates outer product of gradient for BHHH-based Hessian approximate. Returns flattened matrix.
        """
        outer_products = np.einsum('ni, nj->nij', grad, grad)
        return outer_products.reshape((outer_products.shape[0], -1))
    
    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        """
        :param y: observation matrix (n_obs, n_t) -> np.ndarray
        :param y_hat: current (transformed) parameter estimates matrix (n_obs, n_t) -> np.ndarray (e.g. y_hat = log(lambda) for exponential distribution assuming log link)
        :param iteration: current number of iteration, generally not needed
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: IFM-approximated grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix

        note: we assume Normal marginals and a bivariate Gaussian copula with one parameter
        """
        # define and reshape observation and parameter matrices
        y0 = y[:, 0].reshape(-1,1)
        y1 = y[:, 1].reshape(-1,1)
        mu0 = y_hat[:, 0].reshape(-1,1)
        sigma0 = np.exp(y_hat[:, 1]).reshape(-1,1)
        mu1 = y_hat[:, 2].reshape(-1,1)
        sigma1 = np.exp(y_hat[:, 3]).reshape(-1,1)
        rho = self.cop_par_inv_link(y_hat[:, -1]).reshape(-1,1)

        # get gradient for each marginal dist parameters
        mu0_grad = -self.norm_lpdf_mu_grad(y0, mu0, sigma0)
        mu1_grad = -self.norm_lpdf_mu_grad(y1, mu1, sigma1)
        sigma0_grad = -self.norm_lpdf_sigma_grad(y0, mu0, sigma0)*sigma0
        sigma1_grad = -self.norm_lpdf_sigma_grad(y1, mu1, sigma1)*sigma1

        # compute pseudo-observations; rank-based at iteration 0, cdf-based at iteration > 0
        if iteration == 0:
            u = self.rank_pseudo_obs(y)
        else:
            u1 = stats.norm.cdf(y0, loc = mu0, scale = sigma0)
            u2 = stats.norm.cdf(y1, loc = mu1, scale = sigma1)
            u = np.hstack((u1, u2))
        
        # get copula parameter grad
        cop_grad = -self.gaussian_lpdf_grad(u, rho)*self.cop_par_inv_link_grad(self.cop_par_link(rho))

        # combine grad components into a vector
        grad = np.hstack((mu0_grad, sigma0_grad, mu1_grad, sigma1_grad, cop_grad))
       
        # get second derivative for each marginal dist parameters
        mu0_hess = -self.norm_lpdf_mu_hess(sigma0)
        mu1_hess = -self.norm_lpdf_mu_hess(sigma1)
        sigma0_hess = -self.norm_lpdf_sigma_hess(y0, mu0, sigma0)*np.square(sigma0) + sigma0_grad #special case
        sigma1_hess = -self.norm_lpdf_sigma_hess(y1, mu1, sigma1)*np.square(sigma1) + sigma1_grad #special case
        mixed0_hess = -self.norm_lpdf_mixed_hess(y0, mu0, sigma0)*sigma0
        mixed1_hess = -self.norm_lpdf_mixed_hess(y1, mu1, sigma1)*sigma1

        # get copula parameter hess
        cop_hess = -(self.gaussian_lpdf_hess(u, rho)*(self.cop_par_inv_link_grad(self.cop_par_link(rho)))**2 + self.gaussian_lpdf_grad(u, rho)*self.cop_par_inv_link_hess(self.cop_par_link(rho)))

        #combine hess component into a vector
        #hess_diags = np.hstack((mu0_hess, sigma0_hess, mu1_hess, sigma1_hess, cop_hess, mixed0_hess, mixed1_hess, grad))
        hess_diags = np.hstack((mu0_hess, sigma0_hess, mu1_hess, sigma1_hess, cop_hess)) #, mixed0_hess, mixed1_hess, grad))

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
        #diag, off_diag, G = self.hessian_sum_split(H)
        #A = np.diag(diag)
        #A[1, 0] += off_diag[0]
        #A[0, 1] += off_diag[0]
        #A[2, 3] += off_diag[1]
        #A[3, 2] += off_diag[1]
        #H_inv = np.linalg.inv(A)
        #optimal_loss = np.sum(G.T @ H_inv @ G)
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
        #diag, off_diag, _ = self.hessian_sum_split(H)
        #A = np.diag(diag)
        #A[1, 0] += off_diag[0]
        #A[0, 1] += off_diag[0]
        #A[2, 3] += off_diag[1]
        #A[3, 2] += off_diag[1]
        #H_inv = np.linalg.inv(A)
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
        y0 = y[:, 0].reshape(-1,1)
        y1 = y[:, 1].reshape(-1,1)
        mu0 = y_hat[:, 0].reshape(-1,1)
        sigma0 = np.exp(y_hat[:, 1]).reshape(-1,1)
        mu1 = y_hat[:, 2].reshape(-1,1)
        sigma1 = np.exp(y_hat[:, 3]).reshape(-1,1)
        rho = self.cop_par_inv_link(y_hat[:, -1]).reshape(-1,1)
        u = self.rank_pseudo_obs(y)

        #lt0 = stats.norm.logpdf(y0, mu0, sigma0)
        #lt1 = stats.norm.logpdf(y1, mu1, sigma1)
        #print(-np.sum(lt0 + lt1))
        
        l0 = self.norm_lpdf(y0, mu0, sigma0)
        l1 = self.norm_lpdf(y1, mu1, sigma1)
        lc = self.gaussian_lpdf(u, rho)
        #print(-np.sum(l0 + l1 + lc))
        #return -np.sum(l0 + l1 + lc)
        #print(-(l0 + l1))
        return -(l0 + l1 + lc)