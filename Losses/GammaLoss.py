import numpy as np
import mbtr.losses as loss
from mbtr.utils import check_pars
import scipy.stats as stats
import scipy.special as special

class GammaLikelihood(loss.Loss):
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

        :return: np.ndarray with parameter initial guess
        """
        alpha_hat, _, theta_hat = stats.gamma.fit(y, floc=0)
        #shape_fit1, _, scale_fit1 = stats.gamma.fit(y, floc=0)
        omega1_hat = np.log(alpha_hat)
        omega2_hat = -np.log(theta_hat)
        #nat_shape1_hat0 = np.log(shape_fit1) #np.log(shape_fit1 + 1)
        #nat_rate1_hat0 = np.log(1/scale_fit1)
        #initial_parameters = np.array([[nat_shape1_hat0, nat_rate1_hat0]])
        #print("alpha_hat")
        #print(alpha_hat)
        #print("theta_hat")
        #print(theta_hat)
        return np.array([[omega1_hat, omega2_hat]]) # shape = (2,) #initial_parameters#.reshape(1,2) #np.hstack((nat_shape1_hat0, nat_rate1_hat0)) #np.hstack((nat_shape1_hat0*np.ones((y.shape[0], 1)), nat_rate1_hat0*np.ones((y.shape[0], 1)))) #, nat_shape2_hat0*np.ones((y.shape[0], 1)) , nat_rate2_hat0*np.ones((y.shape[0], 1)))) #, np.zeros((y.shape[0], 1))))
    
    #def gamma_lpdf_grad_eta1(self, y, eta1, eta2):
        #return np.log(y) - special.psi(eta1+1) + np.log(eta2) #np.where(y != 0, np.log(y) - special.psi(eta1+1) + np.log(eta2), 0) # if no claim observed i.e. y = 0, gradient = 0

    #def gamma_lpdf_grad_eta2(self, y, eta1, eta2):
        #return -y + (eta1+1)/eta2 #np.where(y != 0, -y + (eta1+1)/eta2, 0) # if no claim observed, gradient = 0
    
    #def gamma_lpdf_hess_eta1(self, eta1):
        #return -special.polygamma(1, eta1+1)
    
    #def gamma_lpdf_hess_eta2(self, eta1, eta2):
        #return -(eta1+1)/(eta2**2)

    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        """
        :param y: observation matrix (n_obs, n_t) -> np.ndarray
        :param y_hat: current parameter estimates matrix (n_obs, n_param) -> np.ndarray (e.g. y_hat = log(lambda) for exponential distribution assuming log link)
        :param iteration: current number of iteration
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix
        """
        #if iteration == 0:
            #natshape1 = np.exp(y_hat[0]) - 1
            #natrate1 = np.exp(y_hat[1])
        #else:
        #natshape1 = (np.exp(y_hat[:, 0])-1).reshape(-1,1)
        #natrate1 = np.exp(y_hat[:, 1]).reshape(-1,1)


        #natshape1 = (np.exp(y_hat[:, 0])-1).reshape(-1,1)
        #natrate1 = np.exp(y_hat[:, 1]).reshape(-1,1)

        #print("y_hat")
        #print(y_hat)

        omega1 = y_hat[:, 0].reshape(-1, 1) #shape = (n_obs, 1)
        omega2 = y_hat[:, 1].reshape(-1, 1) #shape = (n_obs, 1)

        omega1_grad = (special.psi(np.exp(omega1)) - omega2 - np.log(y))*np.exp(omega1) #shape = (n_obs, 1)
        omega2_grad = y*np.exp(omega2) - np.exp(omega1) #shape = (n_obs, 1)

        omega1_hess = 2*special.polygamma(1, np.exp(omega1))*np.exp(2*omega1) #shape = (1, 1) at iteration 1, (n_obs, 1)
        omega2_hess = np.exp(omega1) + y*np.exp(omega2) #shape = (n_obs, 1)


        # compute gradient for each marginal dist parameters
        #natshape1_grad = -self.gamma_lpdf_grad_eta1(y, natshape1, natrate1)*(natshape1+1)
        #natrate1_grad = -self.gamma_lpdf_grad_eta2(y, natshape1, natrate1)*natrate1



        #natrate1_grad = -self.gamma_lpdf_grad_eta2(y1, natshape1, natrate1)*natrate1 # why y1?
        #natshape2_grad = -self.gamma_lpdf_grad_eta1(y0, natshape2, natrate2)*(natshape2+1) # why y0?
        #natshape2_grad = -self.gamma_lpdf_grad_eta1(y1, natshape2, natrate2)*(natshape2+1)
        #natrate2_grad = -self.gamma_lpdf_grad_eta2(y1, natshape2, natrate2)*natrate2 

        # compute hessian for each marginal parameters
        #if iteration == 0:
            #natshape1_hess = -self.gamma_lpdf_hess_eta1(natshape1)*((natshape1+1)**2)*np.ones_like(natshape1_grad)
            #natrate1_hess = -(self.gamma_lpdf_hess_eta2(natshape1, natrate1)*(natrate1**2) - natrate1_grad)*np.ones_like(natrate1_grad) #-self.gamma_lpdf_hess_eta2(natshape1, natrate1)*(natrate1**2)*np.ones_like(natrate1_grad)
        #else:
            #natshape1_hess = -self.gamma_lpdf_hess_eta1(natshape1)*((natshape1+1)**2)
            #natrate1_hess = -(self.gamma_lpdf_hess_eta2(natshape1, natrate1)*(natrate1**2) - natrate1_grad) #-self.gamma_lpdf_hess_eta2(natshape1, natrate1)*(natrate1**2)



        #natshape2_hess = -self.gamma_lpdf_hess_eta1(natshape2)*((natshape2+1)**2)
        #natrate2_hess = -self.gamma_lpdf_hess_eta2(natshape2, natrate2)*(natrate2**2)
        #print(natshape1_grad.reshape(-1,1).shape)
        #print(natrate1_grad.reshape(-1,1).shape)

        # combine gradient components into a vector
        grad = np.hstack((omega1_grad, omega2_grad)) #shape = (n_obs, n_param), in  this case n_param = 2
        #grad = np.hstack((natshape1_grad.reshape(-1,1), natrate1_grad.reshape(-1,1))) #, natshape2_grad, natrate2_grad)) #, cop_grad))

        # combine hessian components into a vector
        hess_diags = np.hstack((omega1_hess*np.ones_like(omega2_hess), omega2_hess)) #np.hstack((np.ones_like(omega2_hess), np.ones_like(omega2_hess)))
        #hess_diags = 2*np.hstack((natshape1_hess.reshape(-1,1), natrate1_hess.reshape(-1,1))) #, natshape2_hess, natrate2_hess)) #, cop_hess))

        #print("grad")
        #print(grad[2, :])
        #print("hess")
        #print(hess_diags[2, :])

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
        optimal_loss = -np.sum(H_inv @ G2)
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
        #natshape1 = (np.exp(y_hat[:, 0])-1).reshape(-1,1)
        #natrate1 = np.exp(y_hat[:, 1]).reshape(-1,1)
        #natshape2 = (np.exp(y_hat[:, 2])-1).reshape(-1,1)
        #natrate2 = np.exp(y_hat[:, 3]).reshape(-1,1)

        omega1 = y_hat[:, 0].reshape(-1, 1) #shape = (n_obs, 1)
        omega2 = y_hat[:, 1].reshape(-1,1) #shape = (n_obs, 1)

        alpha = np.exp(omega1)
        beta = np.exp(omega2)
        
        #l2 = np.sum(np.where(y != 0, stats.gamma.logpdf(x = y, a = (natshape1 + 1), scale = 1/natrate1), 0))
        nll = -np.sum(stats.gamma.logpdf(x = y, a = alpha, loc = 0, scale = 1/beta)) #-np.sum(stats.gamma.logpdf(x = y, a = (natshape1 + 1), scale = 1/natrate1))
        
        return nll