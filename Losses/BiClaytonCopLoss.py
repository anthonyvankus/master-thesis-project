import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats
import scipy.special as special
#import vinecopulas as vine

class BiClaytonCopLikelihood(loss.Loss):
    """
    Probabilistic prediction / Parametric inference using copula-based multivariate negative log-likelihood loss functions.
    """
    required_pars = [] #empty
    #iteration = 0
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

    def clayton_tau2par(self, tau):
        """
        Return Clayton parameters from Kendall's tau values
        """
        return 2*(tau/(1-tau))

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediction. This is loss-specific. We set the dependence parameters to Kendall's tau estimates for the entire dataset of pseudo-observations

        :param y: target matrix of the training set

        :return: np.ndarray with (transformed) parameter initial guess
        """
        u0 = y[:, 0].reshape(-1,1)
        u1 = y[:, 1].reshape(-1,1)
        tau, _ = stats.kendalltau(u0, u1)
        #theta = self.clayton_tau2par(tau) # theta based on kendall tau inversion
        
        return special.logit(tau)*np.ones((y.shape[0], 1)) #np.log(theta)*np.ones((y.shape[0], 1)) # log link as reparametrization

    def clayton_lpdf(self, u, theta):
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)
        
        filt = (theta >= 100)

        u0_filt = u0[filt].reshape(-1,1)
        u1_filt = u1[filt].reshape(-1,1)

        u0_no_filt = u0[~filt].reshape(-1,1)
        u1_no_filt = u1[~filt].reshape(-1,1)

        theta_filt = theta[filt].reshape(-1,1)
        theta_no_filt = theta[~filt].reshape(-1,1)
        
        lpdf = np.zeros_like(theta)
        A = np.ones_like(theta)

        # with large theta filter
        lpdf[filt] = (special.log1p(theta_filt) - (1+theta_filt)*np.log(u0_filt*u1_filt) - (2+1/theta_filt)*np.logaddexp(-theta_filt*np.log(u0_filt), -theta_filt*np.log(u1_filt))).flatten()

        # without large theta filter
        A[~filt] = (self.clayton_Afunc(u[~filt.flatten()], theta[~filt].reshape(-1,1))).flatten()
        lpdf[~filt] = (special.log1p(theta_no_filt) - (1+theta_no_filt)*np.log(u0_no_filt*u1_no_filt) - (2+1/theta_no_filt)*np.log(A[~filt].reshape(-1,1))).flatten()

        return lpdf.reshape(-1,1)

    def clayton_Afunc(self, u, theta):
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)
        
        return u0**(-theta) + u1**(-theta) - 1
    
    def clayton_Afunc_grad(self, u, theta):
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)

        return -((u0**(-theta))*np.log(u0) + (u1**(-theta))*np.log(u1))
    
    def clayton_Afunc_hess(self, u, theta):
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)

        return (u0**(-theta))*(np.log(u0)**2) + (u1**(-theta))*(np.log(u1)**2)

    def clayton_logA_grad(self, u, theta):
        A = self.clayton_Afunc(u, theta)
        return self.clayton_Afunc_grad(u, theta)/A
    
    def clayton_logA_hess(self, u, theta):
        A = self.clayton_Afunc(u, theta)
        dlogA = self.clayton_logA_grad(u, theta)
        return (self.clayton_Afunc_hess(u, theta)/A) - (dlogA**2)

    def clayton_lpdf_grad(self, u, theta):
        """
        Clayton copula gradient w.r.t dependence parameter theta

        :param u: bivariate pseudo-observation vector (n_obs, 2)
        :param theta: current dependence parameter estimates (n_obs, 1)
        
        :return: Clayton copula lpdf gradient
        """
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)

        filt = (theta >= 100)

        u0_filt = u0[filt].reshape(-1,1)
        u1_filt = u1[filt].reshape(-1,1)

        u0_no_filt = u0[~filt].reshape(-1,1)
        u1_no_filt = u1[~filt].reshape(-1,1)

        min_u_filt = np.min(u[filt.flatten()], axis = 1).reshape(-1,1)
        max_u_filt = np.max(u[filt.flatten()], axis = 1).reshape(-1,1)

        theta_filt = theta[filt].reshape(-1,1)
        theta_no_filt = theta[~filt].reshape(-1,1)
        
        A = np.ones_like(theta)
        logA = np.zeros_like(theta)
        dlogA = np.zeros_like(theta)
        lpdf_grad = np.zeros_like(theta)

        # with large theta filter
        logA[filt] = (np.logaddexp(-theta_filt*np.log(u0_filt), -theta_filt*np.log(u1_filt))).flatten()
        dlogA[filt] = (-np.log(min_u_filt) + np.log(min_u_filt/max_u_filt)*special.expit(theta_filt*np.log(min_u_filt/max_u_filt))).flatten()
        lpdf_grad[filt] = (1/(1+theta_filt) - np.log(u0_filt*u1_filt) - (2 + 1/theta_filt)*dlogA[filt].reshape(-1,1) + logA[filt].reshape(-1,1)/(theta_filt**2)).flatten()

        # without large theta filter
        A[~filt] = (self.clayton_Afunc(u[(~filt).flatten()], theta_no_filt)).flatten()
        logA[~filt] = np.log(A[~filt].reshape(-1,1)).flatten()
        dlogA[~filt] = self.clayton_logA_grad(u[(~filt).flatten()], theta_no_filt).flatten()
        lpdf_grad[~filt] = (1/(1+theta_no_filt) - np.log(u0_no_filt * u1_no_filt) - (2 + 1/theta_no_filt)*dlogA[~filt].reshape(-1,1) + logA[~filt].reshape(-1,1)/np.square(theta_no_filt)).flatten()

        return lpdf_grad.reshape(-1,1) #1/(1+theta) - np.log(u0*u1) + (-2-(1/theta))*dlogA + logA/np.square(theta) #(np.log(A)/(theta**2)) + (((-2-(1/theta))*dA)/A) - np.log(u0*u1)
    
    def clayton_lpdf_hess(self, u, theta):
        """
        Clayton copula lpdf hessian w.r.t dependence parameter theta

        :param u: bivariate pseudo-observation vector (n_obs, 2)
        :param theta: current dependence parameter estimates (n_obs, 1)
        
        :return: Clayton copula lpdf hessian
        """
        u0 = u[:, 0].reshape(-1,1)
        u1 = u[:, 1].reshape(-1,1)
        
        filt = (theta >= 100)

        u0_filt = u0[filt].reshape(-1,1)
        u1_filt = u1[filt].reshape(-1,1)

        #u0_no_filt = u0[~filt].reshape(-1,1)
        #u1_no_filt = u1[~filt].reshape(-1,1)

        min_u_filt = np.min(u[filt.flatten()], axis = 1).reshape(-1,1)
        max_u_filt = np.max(u[filt.flatten()], axis = 1).reshape(-1,1)

        theta_filt = theta[filt].reshape(-1,1)
        theta_no_filt = theta[~filt].reshape(-1,1)
        
        A = np.ones_like(theta)
        logA = np.zeros_like(theta)
        dlogA = np.zeros_like(theta)
        d2logA = np.ones_like(theta)
        lpdf_hess = np.zeros_like(theta)

        # with large theta filter
        logA[filt] = (np.logaddexp(-theta_filt*np.log(u0_filt), -theta_filt*np.log(u1_filt))).flatten()
        dlogA[filt] = (-np.log(min_u_filt) + np.log(min_u_filt/max_u_filt)*special.expit(theta_filt*np.log(min_u_filt/max_u_filt))).flatten()
        d2logA[filt] = (np.log(min_u_filt/max_u_filt)*special.expit(theta_filt*np.log(min_u_filt/max_u_filt))*(1 - special.expit(theta_filt*np.log(min_u_filt/max_u_filt)))).flatten()
        lpdf_hess[filt] = (-1/((1+theta_filt)**2) - (2 + 1/theta_filt)*d2logA[filt].reshape(-1,1) + 2*dlogA[filt].reshape(-1,1)/(theta_filt**2) - 2*logA[filt].reshape(-1,1)/(theta_filt**3)).flatten()

        # without large theta filter
        A[~filt] = (self.clayton_Afunc(u[(~filt).flatten()], theta_no_filt)).flatten()
        logA[~filt] = np.log(A[~filt].reshape(-1,1)).flatten()
        dlogA[~filt] = self.clayton_logA_grad(u[(~filt).flatten()], theta_no_filt).flatten()
        d2logA[~filt] = self.clayton_logA_hess(u[(~filt).flatten()], theta_no_filt).flatten()
        lpdf_hess[~filt] = (-1/((1+theta_no_filt)**2) - (2 + 1/theta_no_filt)*d2logA[~filt].reshape(-1,1) + 2*dlogA[~filt].reshape(-1,1)/(theta_no_filt**2) - 2*logA[~filt].reshape(-1,1)/(theta_no_filt**3)).flatten()

        #logA = np.log(self.clayton_Afunc(u, theta))
        #dlogA = self.clayton_logA_grad(u, theta)
        #d2logA = self.clayton_logA_hess(u, theta)

        return lpdf_hess.reshape(-1,1) #(-1/((1+theta)**2)) + (-2-(1/theta))*d2logA + (2/(theta**2))*dlogA - (2/(theta**3))*logA #dA**2 + dA*np.log(u0) + a + b - c - dlc**2
    
    #def clayton_rho(self, theta):
        a = 1/((3*theta-2)*(2*theta-1))
        b = 1/(2*(theta-1))
        return a*(1 + theta*b*(special.polygamma(1, b) - special.polygamma(1, theta/(2*(theta-2)))) + b*(special.polygamma(1, theta*b) - special.polygamma(1, (2*theta-1)/(2*(theta-2)))))

    #def clayton_fisher_information(self, theta):
        rho = self.clayton_rho(theta)
        return 1/np.square(theta) + 2/(theta*(theta-1)*(2*theta-1)) + (4*theta)/(3*theta-2) - ((2*(2*theta-1))/(theta-1))*rho
    
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
        #print(self.tree_loss(u, y_hat))
        #self.iteration = iteration
        #define level of tolerance for hessian and gradient value (to avoid hessian -> 0 and grad -> -inf)
        #tol_hess = -0.15
        #tol_grad = 1e20
        #print(np.min(u0))
        #print(np.min(u1))

        # define current dependence parameter
        #theta = np.exp(y_hat[:, 0]).reshape(-1,1)
        tau = special.expit(y_hat)
        theta = self.clayton_tau2par(tau)
        #print(np.max(theta))
        #print(np.min(theta))

        # get copula parameter gradient
        #cop_grad = self.clayton_lpdf_grad(u, theta)#*theta
        cop_grad = self.clayton_lpdf_grad(u, theta)*(2/(1-tau)**2)
        #cop_grad[cop_grad > tol_grad] = tol_grad # gradient clipping
        #print(np.max(cop_grad))

        # get copula parameter hessian
        #cop_hess = self.clayton_lpdf_hess(u, theta)*(4/(1-tau)**4) + self.clayton_lpdf_grad(u, theta)*(4/(1-tau)**3) #-(self.clayton_lpdf_hess(u, theta)*(theta**2) + self.clayton_lpdf_grad(u, theta)*theta)
        #print(np.max(cop_hess))
        #print(np.min(cop_hess))
        #cop_hess[cop_hess > tol_hess] = tol_hess
        #print(np.max(cop_hess))

        # combine grad components into a vector
        grad = -cop_grad*(tau*(1-tau)) #-cop_grad*theta

        #combine hess component into a vector
        hess_diags = grad**2 #-cop_hess*((tau*(1-tau))**2) - cop_grad*(tau*(1-tau)*(1-2*tau)) #grad**2 #-cop_hess*(theta**2) + grad #cop_grad**2 #-cop_hess*(theta**2) + grad #cop_grad**2 #-(np.ones((y.shape[0], 1))*(theta**2) + self.clayton_lpdf_grad(u, theta)*theta) #cop_hess #cop_grad**2 #np.ones((y.shape[0], 1)) #-self.clayton_fisher_information(theta)*np.square(theta)

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
        #if self.iteration >= 30:
        #    optimal_loss = 2*optimal_loss
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
        #if self.iteration >= 30:
        #    optimal_response = 2*optimal_response
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
        #theta = np.exp(y_hat[:, 0]).reshape(-1,1)
        tau = special.expit(y_hat)
        theta = self.clayton_tau2par(tau)

        return -np.sum(self.clayton_lpdf(u, theta))