import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats
import scipy.special as special

class PricingLikelihood(loss.Loss):
    """
    Probabilistic prediction / Parametric inference using copula-based multivariate negative log-likelihood loss functions.
    """
    required_pars = ['warmup'] # might expand subclass later to accept copula and marginal families as parameters
    
    """"
    :param lambda_weights:  quadratic penalization parameter for the leaves weights
    :param lambda_leaves: quadratic penalization parameter for the number of leaves
    :param loss_kwargs: additional parameters needed for a specific loss type
    """
    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.warmup = self.kwargs['warmup']
        self.n_dims = 7

    def set_dimension(self, n_dims):
        pass

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the marginal parameters to global MLE estimates and assume independence i.e. copula parameter = 0.

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess i.e. (r, logodds, natshape1, natrate1, natshape2, natrate2, rho)
        """
        expo = y[:, -1].reshape(-1,1)
        n = y[:, 0].reshape(-1,1)
        t = y[:, 1].reshape(-1,1)
        y1 = y[:, 2].reshape(-1,1)
        y2 = y[:, 3].reshape(-1,1)
        r_hat0 = np.log(np.sum(n, axis = 0)/np.sum(expo, axis = 0))
        p0 = np.sum(t)/np.sum(n)
        logodds_hat0 = special.logit(p0)
        shape_fit1, _, scale_fit1 = stats.gamma.fit(y1[np.nonzero(y1)], floc=0)
        shape_fit2, _, scale_fit2 = stats.gamma.fit(y2[np.nonzero(y2)], floc=0)
        nat_shape1_hat0 = np.log(shape_fit1 + 1)
        nat_rate1_hat0 = np.log(1/scale_fit1)
        nat_shape2_hat0 = np.log(shape_fit2 + 1)
        nat_rate2_hat0 = np.log(1/scale_fit2)
        return np.hstack((r_hat0*np.ones((y.shape[0], 1)), logodds_hat0*np.ones((y.shape[0], 1)), nat_shape1_hat0*np.ones((y.shape[0], 1)), nat_rate1_hat0*np.ones((y.shape[0], 1)), nat_shape2_hat0*np.ones((y.shape[0], 1)) , nat_rate2_hat0*np.ones((y.shape[0], 1)), np.zeros((y.shape[0], 1))))
    
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
    
    def poisson_lpdf_grad(self, n, r, expo):
        return n - expo*np.exp(r)
    
    def poisson_lpdf_hess(self, r, expo):
        return -expo*np.exp(r)
    
    def binomial_lpdf_grad(self, n, t, logodds):
        return t - n*special.expit(logodds)
    
    def binomial_lpdf_hess(self, n, logodds):
        return - np.where(n != 0, n*np.exp(logodds)/(1+np.exp(logodds))**2, 1)
    
    def gamma_lpdf_grad_eta1(self, y, eta1, eta2):
        return np.where(y != 0, np.log(y) - special.psi(eta1+1) + np.log(eta2), 0)

    def gamma_lpdf_grad_eta2(self, y, eta1, eta2):
        return np.where(y != 0, -y + (eta1+1)/eta2, 0)
    
    def gamma_lpdf_hess_eta1(self, eta1):
        return -special.polygamma(1, eta1+1)
    
    def gamma_lpdf_hess_eta2(self, eta1, eta2):
        return -(eta1+1)/(eta2**2)
    
    def gaussian_lpdf(self, u, rho):
        a = stats.norm.ppf(u[:, 0]).reshape(-1,1)
        b = stats.norm.ppf(u[:, 1]).reshape(-1,1)
        c = a*b
        d = a**2 + b**2
        return np.sum(-np.log(np.sqrt(1-rho**2)) - ((rho**2)*d - 2*rho*c)/(2*(1-rho**2))) #np.sum(np.where(a == -np.inf or b == -np.inf, 0, -np.log(np.sqrt(1-rho**2)) - ((rho**2)*d - 2*rho*c)/(2*(1-rho**2))))

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
        return (rho/(1-rho**2) + ((1+rho**2)*c - d*rho)/((1-rho**2)**2)) #np.where(a == -np.inf or b == -np.inf, 0, (rho/(1-rho**2) + ((1+rho**2)*c - d*rho)/((1-rho**2)**2)))
    
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
        return ((1+rho**2) + 2*c*rho - d)/((1-rho**2)**2) + (4*rho*(c*(1+rho**2) - d*rho))/((1-rho**2)**3) #np.where(a == -np.inf or b == -np.inf, 1, ((1+rho**2) + 2*c*rho - d)/((1-rho**2)**2) + (4*rho*(c*(1+rho**2) - d*rho))/((1-rho**2)**3))
    
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
        :param iteration: current number of iteration, needed to indicate when warm-up phase ends
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix
        """
        n_obs = y.shape[0]
        # define and reshape observation and parameter matrices
        n = y[:, 0].reshape(-1,1)
        t = y[:, 1].reshape(-1,1)
        y0 = y[:,2].reshape(-1,1)
        y1 = y[:,3].reshape(-1,1)
        expo = y[:,4].reshape(-1,1)

        r = y_hat[:, 0].reshape(-1,1)
        logodds =  y_hat[:, 1].reshape(-1,1)
        natshape1 = (np.exp(y_hat[:, 2])-1).reshape(-1,1)
        natrate1 = np.exp(y_hat[:, 3]).reshape(-1,1)
        natshape2 = (np.exp(y_hat[:, 4])-1).reshape(-1,1)
        natrate2 = np.exp(y_hat[:, 5]).reshape(-1,1)

        # compute gradient for each marginal dist parameters
        r_grad = -self.poisson_lpdf_grad(n, r, expo)
        logodds_grad = -self.binomial_lpdf_grad(n, t, logodds)
        natshape1_grad = -self.gamma_lpdf_grad_eta1(y0, natshape1, natrate1)*(natshape1+1)
        natrate1_grad = -self.gamma_lpdf_grad_eta2(y1, natshape1, natrate1)*natrate1
        natshape2_grad = -self.gamma_lpdf_grad_eta1(y0, natshape2, natrate2)*(natshape2+1)
        natrate2_grad = -self.gamma_lpdf_grad_eta2(y1, natshape2, natrate2)*natrate2

        # compute hessian for each marginal parameters
        r_hess = -self.poisson_lpdf_hess(r, expo)
        logodds_hess = -self.binomial_lpdf_hess(n, logodds)
        natshape1_hess = -self.gamma_lpdf_hess_eta1(natshape1)*((natshape1+1)**2)
        natrate1_hess = -self.gamma_lpdf_hess_eta2(natshape1, natrate1)*(natrate1**2)
        natshape2_hess = -self.gamma_lpdf_hess_eta1(natshape2)*((natshape2+1)**2)
        natrate2_hess = -self.gamma_lpdf_hess_eta2(natshape2, natrate2)*(natrate2**2)

        if iteration >= self.warmup:
            # define current dependence parameter
            rho = self.cop_par_inv_link(y_hat[:, -1]).reshape(-1,1)

            # compute cdf-based pseudo-observations
            u1 = stats.gamma.cdf(y0, a = natshape1 + 1, scale = 1/natrate1) 
            u2 = stats.gamma.cdf(y1, a = natshape2 + 1, scale = 1/natrate2)
            u = np.hstack((u1, u2))

            # get copula parameter gradient
            cop_grad = np.zeros((n_obs, 1))
            mask = ~np.any(u == 0, axis = 1)
            cop_grad[mask] = -self.gaussian_lpdf_grad(u[mask], rho[mask])*self.cop_par_inv_link_grad(self.cop_par_link(rho[mask]))

            # get copula parameter hessian
            cop_hess = np.ones((n_obs, 1))
            cop_hess[mask] = -(self.gaussian_lpdf_hess(u[mask], rho[mask])*(self.cop_par_inv_link_grad(self.cop_par_link(rho[mask])))**2 + self.gaussian_lpdf_grad(u[mask], rho[mask])*self.cop_par_inv_link_hess(self.cop_par_link(rho[mask])))
        else:
            # no parameter inference done before warm-up phase ends
            cop_grad = np.zeros((y.shape[0], 1))
            cop_hess = np.ones((y.shape[0], 1))

        # combine gradient components into a vector
        grad = np.hstack((r_grad, logodds_grad, natshape1_grad, natrate1_grad, natshape2_grad, natrate2_grad, cop_grad))

        # combine hessian components into a vector
        hess_diags = np.hstack((r_hess, logodds_hess, natshape1_hess, natrate1_hess, natshape2_hess, natrate2_hess, cop_hess))

        return grad, hess_diags
    
    def H_inv(self, H):
        """
        Computes the inverse of the Hessian, given the Hessian's diagonal of the current leave.

        :param H: current leaf Hessian's diagonal (n_t, 1)
        """
        H[-1] = np.abs(H[-1]) # (Dauphin et al., 2014) saddle-free Hessian computation requires we take the absolute value of the hessian
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
        n = y[:, 0].reshape(-1,1)
        t = y[:, 1].reshape(-1,1)
        y0 = y[:, 2].reshape(-1,1)
        y1 = y[:, 3].reshape(-1,1)
        expo = y[:, 4].reshape(-1,1)

        r = y_hat[:, 0].reshape(-1,1)
        logodds =  y_hat[:, 1].reshape(-1,1)
        natshape1 = (np.exp(y_hat[:, 2])-1).reshape(-1,1)
        natrate1 = np.exp(y_hat[:, 3]).reshape(-1,1)
        natshape2 = (np.exp(y_hat[:, 4])-1).reshape(-1,1)
        natrate2 = np.exp(y_hat[:, 5]).reshape(-1,1)
        rho = self.cop_par_inv_link(y_hat[:, -1]).reshape(-1,1)
        #u = np.where(y0 != 0 and y1!= 0, self.rank_pseudo_obs(np.hstack((y0, y1))), np.zeros((0,0)))
        # compute cdf-based pseudo-observations
        u1 = stats.gamma.cdf(y0, a = natshape1 + 1, scale = 1/natrate1) 
        u2 = stats.gamma.cdf(y1, a = natshape2 + 1, scale = 1/natrate2)
        u = np.hstack((u1, u2))
        mask = ~np.any(u == 0, axis = 1)
        
        l0 = np.sum(stats.poisson.logpmf(k = n, mu = expo*np.exp(r)))
        l1 = np.sum(np.where(n != 0, stats.binom.logpmf(k = t, n = n, p = special.expit(logodds)), 0))
        l2 = np.sum(np.where(y0 != 0, stats.gamma.logpdf(x = y0, a = (natshape1 + 1), scale = 1/natrate1), 0))
        l3 = np.sum(np.where(y1 != 0, stats.gamma.logpdf(x = y1, a = (natshape2 + 1), scale = 1/natrate2), 0))
        lc = self.gaussian_lpdf(u[mask], rho[mask])
        
        return -(l0 + l1 + l2 + l3 + lc)