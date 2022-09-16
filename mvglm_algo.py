import numpy as np
from scipy.linalg import block_diag, sqrtm
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from multimodal.datasets.data_sample import DataSample, MultiModalArray
from multimodal.kernels.mkernel import MKernel

"""
Usage:
    create a MVGLM object via:
        mvglm = MVGLM(lmbda, gamma, learn_w, model, kernel, kernel_params)
    learn the model:
        mvglm.fit(x_train, y_train)
    predict with the model:
        predictions = mvglm.predict(x_test)
    calculate Cook's distances:
        cooks = mvglm.cooks_distance()
"""

class MVGLM(MKernel, BaseEstimator, ClassifierMixin, RegressorMixin):

    def __init__(self, lmbda=0.1, gamma=0.1, learn_w=True, n_loops=20, 
            verbose=True, model='logistic', CC=False, check_convergence=True, 
            wPen='L2', kernel='linear', kernel_params=None, condition=False):
        super().__init__()

        """
        :param lmbda: regularisation parameter for alpha
        :param gamma: regularisation parameter for w
        :param learn_w: whether to learn the weight vector
        :param n_loops: maximum number of iterations
        :param verbose: whether to print progress
        :param model: exponential family model; currently supported are
                        ["logistic", "poisson" and "multinomial"]
        :param CC: whether to use the cross-covariance multi-view kernel
        :param check_convergence: whether to check convergence
        :param wPen: the regularisation type for the weight vector; 
                        ["L1", "L2"] are supported.
        :param kernel: list of kernel types, one per view
        :param kernel_params: list of kernel parameter dicts
        :param condition: whether to account for ill-conditioned Gram matrix
        """

        self.lmbda = lmbda
        self.gamma = gamma
        self.learn_w = learn_w
        self.n_loops = n_loops
        self.verbose = verbose
        self.model = model
        self.CC = CC
        self.check_convergence = check_convergence
        self.wPen = wPen
        self.kernel=kernel
        self.kernel_params = kernel_params
        self.condition = condition

    def fit(self, X, labels, views_ind=None):

        """
        :param X: training data of class MultiModalArray or numpy array
        :param labels: array of response labels
        :param views_ind: the indices of each view if numpy array used for X
        """

        self.X_, self.K_ = self._global_kernel_transform(X, 
                                                         views_ind=X.views_ind)

        self.n = self.K_.shape[0]
        self.kernels = self.K_._todict()
        self.views = len(self.kernels)
        
        if len(labels.shape) == 1:
            self.Y = labels[:,None]
        else:
            self.Y = labels
        n = self.n
        if self.model == 'multinomial':
            self.Y = self.Y + 1
        lmbda = self.lmbda
        gamma = self.gamma
        views = self.views  
        self.tol = 1e-4

        learn_w = self.learn_w 
        n_loops = self.n_loops 
        verbose = self.verbose 
        model = self.model 
        CC = self.CC 
        check_convergence = self.check_convergence 
        wPen = self.wPen 

        # ========= initialize Gram Matrix =========
        K = np.zeros((views*n, views*n))
        if CC:
            H = np.eye(n) - np.ones((n,n))/n
            for v in range(views):
                for vv in range(v, views):
                    K[v*n : (v+1)*n, vv*n : (vv+1)*n] = np.matmul(np.matmul(
                        self.kernels[v], H), 
                        self.kernels[vv])/n
                    if v != vv:
                        K[vv*n : (vv+1)*n, v*n : (v+1)*n] = K[
                                v*n : (v+1)*n, vv*n : (vv+1)*n
                                ].T
        else:
            for v in range(views):
                K[v*n : (v+1)*n, v*n : (v+1)*n] = self.kernels[v]
                
        if self.condition:
            K += 0.000001*np.eye(K.shape[0])

        V = np.hstack((np.kron(np.eye(views), np.ones((n,1))), K))
        U = np.vstack((np.zeros((views, n*views + views)),
                       np.hstack((np.zeros((n*views, views)), K))
                       ))
                
        # ========= allocate w, alpha =========
        w = ((1/views)*np.ones((views, 1))
             + np.random.RandomState(123).normal(0,.0001, views)[:,None])
        if model == 'multinomial':
            classes = len(set(self.Y[:,0]))
            self.classes = classes
            self.alpha = [np.zeros((views*n + views, classes)),
                          np.zeros((views*n + views, classes))]
        else:
            classes = 2
            self.classes = classes
            self.alpha = [np.zeros((views*n + views, 1)), 
                          np.zeros((views*n + views, 1))]

        if check_convergence:
            deviance = np.inf
            self.deviances = np.zeros((n_loops, ))
        if learn_w:
            self.ws = np.zeros((n_loops, views))
        
        divide_error = False

        # ========= learn parameters ==========================================
        loop_counter = 0
        while True:
            w_mat = np.kron(w.T, np.eye(n))
            wV = np.matmul(w_mat, V)

            # ========= update alpha ==========================================
            # Newton-Raphson update for alpha
            alpha_prev = self.alpha[1].copy()
            alpha = self.alpha[1].copy()
            for c in range(classes-1):
                eta = np.matmul(wV, alpha)

                # Can evaluate W-inv[y-mu] without inverse or matrix product
                # check for a numerical instability
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        p, Wdiag, W_diff = self._link_fun(
                                                    eta, c, model, 'alpha'
                                                    )
                    except Warning:
                        print('Divide error')
                        divide_error = True
                        break
                z = eta[:,c][:, None] + W_diff

                try:
                    alpha[:, c] = np.linalg.solve( 
                            np.matmul(Wdiag.T*wV.T, wV) + lmbda*U,
                            np.matmul(Wdiag.T*wV.T, z)
                            )[:,0]
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        print('Singular matrix')
                        alpha[:, c] = np.linalg.lstsq( 
                                np.matmul(Wdiag.T*wV.T, wV) + lmbda*U,
                                np.matmul(Wdiag.T*wV.T, z)
                                )[0][:,0]
                    else:
                        raise

            if divide_error:
                self.alpha[1] = self.alpha[0].copy()
                if learn_w:
                    w = self.ws[loop_counter-2,:,None]
                break
            self.alpha[1] = alpha.copy()
            self.alpha[0] = alpha_prev.copy()
            self.eta = eta

            
            # ========= check convergence =====================================
            if loop_counter > 0:
                # convergence criteria
                if check_convergence:
                    converged, deviance = self._convergence_check(self.eta, 
                            deviance, model)
                    self.deviances[loop_counter-1] = deviance
                    if converged or loop_counter >= n_loops-1:
                        self.deviances = self.deviances[:loop_counter-1]
                        break
                else:
                    if loop_counter >= n_loops-1:
                        break
                    

            # ========= update w ==============================================
            if learn_w:
                # Construct Z columnwise from the elements of Va
                Z = np.zeros((self.alpha[1].shape[1], n, views))
                Va = np.matmul(V, self.alpha[1])
                for c in range(self.alpha[1].shape[1]):
                    for v in range(views):
                        Z[c, :, v] = Va[v*n : (v+1)*n, c]
                # If multinomial, different Z for each class. 
                # If not squeeze out empty dimension
                eta = np.matmul(Z,w).squeeze(2).T

                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        p, Wdiag2, W_diff2 = self._link_fun(
                                                  eta, self.alpha[1].shape[1],
                                                  model, 'w'
                                                  )
                    except Warning:
                        print('Divide error')
                        divide_error = True
                if divide_error:
                    self.alpha[1] = self.alpha[0].copy()
                    w = self.ws[loop_counter-2,:,None]
                    break
                else:
                    # If multinomial, ZtWZ is the sum of each ZtWZ
                    ZtWZ_sum = np.sum(
                        np.array([
                            np.matmul(Z[c,:,:].T*Wdiag2[:,c], Z[c,:,:])
                                for c in range(self.alpha[1].shape[1])
                            ]), axis = 0
                    )

                    u = eta + W_diff2

                    if wPen == 'L1':
                        hubr = 1 + (w**2/((1e-3)**2))
                        wGrad = (w * hubr**(-.5)) / 1e-3
                        wHess_diag = hubr**(-1.5)/1e-3
                        v = gamma*wHess_diag*w + wGrad
                    elif wPen == 'L2':
                        v = 0
                        wHess_diag = np.ones_like(w)
                    else:
                        print('wPen incorrectly specified')
                        break
                    # Add regularisation term
                    ZtWZ_sum += gamma*np.diag(wHess_diag[:,0]) 

                    ZtWu = np.sum(
                        np.array([
                            np.matmul( Z[c, :, :].T * Wdiag2[:,c], u[:,c] ) 
                                for c in range(self.alpha[1].shape[1])
                            ]), axis = 0
                    )[:,None]

                    try:
                        w = np.linalg.solve(ZtWZ_sum, ZtWu + v)
                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            print('Singular matrix')
                            w, _, _, _  = np.linalg.lstsq(ZtWZ_sum, ZtWu + v)
                        else:
                            raise

                    self.ws[loop_counter, :] = w.squeeze()
            
            loop_counter += 1
            if verbose:
                print('Done {}/{}'.format(loop_counter,n_loops))
        
        if model=='multinomial':
            self.alpha = self.alpha[1]
        else:
            self.alpha = alpha
        self.w = w
        self._V = V
        self.K = K

        return self

    def predict(self, X, proba=True):

        """
        :param X: test data of class MultiModalArray
        :return: (regression) predictions, array of size test_samples*1
        """
        X, test_kernels = self._global_kernel_transform(
                                X, views_ind=self.X_.views_ind, Y=self.X_
                                )
        test_kernels = test_kernels._todict()

        t = test_kernels[0].shape[0]
        n = self.n
        views = self.views
        
        if self.CC:
            X = np.zeros((views*t, views*n))
            H = np.eye(n) - np.ones((n, n))/n
            for v in range(views):
                for vv in range(views):
                    X[v*t : (v+1)*t, vv*n : (vv+1)*n] = np.matmul(
                            np.matmul(test_kernels[v], H), self.kernels[vv]
                            )/n
#            X = np.hstack((np.ones((t*views, 1)), X))
            X = np.hstack((np.kron(np.eye(views), np.ones((t,1))), X))
            w_mat = np.kron(self.w.T, np.eye(t))
            
            eta = np.matmul(np.matmul(w_mat, X), self.alpha)
        
        else:                    
            w_mat = np.kron(self.w.T, np.eye(t))
            K = np.zeros((views*t, views*n))
            for v in range(views):
                K[v*t : (v+1)*t, v*n : (v+1)*n] = test_kernels[v]
            V = np.hstack((np.kron(np.eye(views), np.ones((t,1))), K))

            eta = np.matmul(np.matmul(w_mat, V), self.alpha)

        if proba:
            p = self._link_fun(eta, self.classes, self.model, 'pred')
            return p
        else:
            return eta


    def _link_fun(self, eta_mat, c, model, update_step):
        
        """
        For alpha update, c is actual class being updated in multinomial model
        For w update, c is the number of classes
        """
        # TODO: could probably remove the need for update_step condition
        
        if update_step == 'alpha':
            if model == 'logistic':
                p = np.exp(eta_mat)/(1+np.exp(eta_mat))
                W = (np.exp(eta_mat)/( (1+np.exp(eta_mat))**2 ))
                W_diff = (1/W) * ((self.Y == c+1).astype(int) - p)
            elif model == 'poisson':
                p = np.exp(eta_mat)
                W = np.exp(eta_mat)
                W[W==0] = 1e-50
                W_diff = (1/W) * (self.Y - p)
            elif model == 'multinomial':
                p = (np.exp(eta_mat[:,c] - eta_mat.max()) 
                        / (np.sum(np.exp(eta_mat - eta_mat.max()), axis=1))
                        )[:, None]
                W = (p*(1 - p))
                W[W==0] = 1e-50
                W_diff = (1/W) * ((self.Y == c+1).astype(int) - p)
            return(p, W, W_diff)

        elif update_step == 'w':
            if model == 'logistic':
                p = np.exp(eta_mat)/(1+np.exp(eta_mat))
                W = (np.exp(eta_mat)/( (1+np.exp(eta_mat))**2 ))
                W_diff = (1/W) * ((self.Y == 1).astype(int) - p)
            elif model == 'poisson':
                p = np.exp(eta_mat)
                W = np.exp(eta_mat) #np.diag( np.exp(eta_mat)[:,0] )
                W[W==0] = 1e-50
                W_diff = (1/W) * (self.Y - p)
            elif model == 'multinomial':
                p = np.array(
                    [(
                        np.exp(eta_mat[:,i] - eta_mat.max()) 
                        / (np.sum(np.exp(eta_mat - eta_mat.max()), axis=1))
                     )[:, None] for i in range(c)
                    ]).squeeze().T
                W = p*(1-p)
                W[W==0] = 1e-50
                W_diff = (1/W)*(np.array(
                    [(self.Y == i+1).astype(int) for i in range(c)]
                    ).squeeze().T - p)
            
            return(p, W, W_diff)

        elif update_step == 'pred':
            if model == 'logistic':
                p = np.exp(eta_mat)/(1+np.exp(eta_mat))
            elif model == 'poisson':
                p = np.exp(eta_mat)
            elif model == 'multinomial':
                p = np.array(
                    [(
                        np.exp(eta_mat[:,i] - eta_mat.max()) 
                        / (np.sum(np.exp(eta_mat - eta_mat.max()), axis=1))
                     )[:, None] for i in range(c)
                    ]).squeeze().T
            return p


    def _convergence_check(self, eta, deviance, model):
        converged = False
        Y = self.Y
        if model == 'logistic':
            p = np.exp(eta)/(1+np.exp(eta))

            dev_new = 2*(
                    np.dot(Y.T, 
                           np.where(
                               Y > 0, np.log(Y/p, where=Y>0), 0)
                          )
                    + np.dot((eta - Y).T,
                             np.log((eta - Y)/(eta - p))
                            )
                    )

        if model == 'poisson':
            p = np.exp(eta)
            dev_new = 2*(np.dot(Y.T, np.log(Y/p))
                         - np.sum(Y - p))

        if model == 'multinomial':
            p = np.array(
                [(
                    np.exp(eta[:,i] - eta.max())
                    / (np.sum( np.exp(eta- eta.max()), axis=1 ))
                 )[:, None] for i in range(self.classes)
                ]).squeeze().T

            dev_mat = np.array(
                    [np.dot((Y == i+1).astype(int).T,
                            np.where((Y == i+1).astype(int) > 0,
                                     np.log(
                                         (Y == i+1).astype(int)/p[:,i,None],
                                         where=(Y == i+1).astype(int)>0
                                         ),
                                     0)
                            ) for i in range(eta.shape[1])]
                    ).squeeze()

            dev_new = 2*dev_mat.sum()

        diff = np.abs( dev_new - deviance )
        if diff < self.tol:
            converged = True

        return converged, dev_new

    def cooks_distance(self):
        n = self.n
        classes = self.classes
        views = self.views

        if self.model == 'multinomial':
            # parameters estimated separately using CD inner-loop. Reshape 
            Y_full = np.array([
                (self.Y == i+1).astype(int) for i in range(classes-1)
                ]).squeeze().T.reshape(((classes-1)*n, 1))
            alpha_full = self.alpha[:,:-1].reshape(
                    ((n+1)*views*(classes-1), 1)
                    )
            wV = np.matmul( np.kron(self.w.T, np.eye(n*(classes-1))),
                    np.kron(self._V, np.eye(classes-1)))
            eta_full = np.matmul(wV, alpha_full)

            
            eta_mat = np.hstack(
                    (eta_full.reshape((n, classes-1)), np.zeros((n, 1)))
                    )
            p = np.array([
                (
                    np.exp(eta_mat[:,i] - eta_mat.max())
                    / (np.sum(np.exp(eta_mat - eta_mat.max()), axis=1))
                )[:, None] for i in range(classes-1)
                ]).squeeze().T 
            U_full = np.vstack(
                    (np.zeros((views*(classes-1), (n+1)*views*(classes-1))),
                     np.hstack(
                         (np.zeros((n*views*(classes-1), views*(classes-1))),
                          np.kron(self.K, np.eye(classes-1)))
                         )
                    )
            )

            # weight matrix is now block diagonal.
            # Form each block then construct
            mat_list = [
                    np.matmul(p[i,:, None], -p[i,:,None].T) for i in range(n)
                    ]
            for i,x in zip(range(n), mat_list):
                    np.fill_diagonal(x, p[i, :, None]*(1-p[i, :, None])) 
            W_mat = block_diag(*mat_list)
            wHalf = np.matmul(sqrtm(W_mat), wV)
            H = np.matmul(
                    wHalf,
                    np.linalg.solve(np.matmul(wV.T, np.matmul(W_mat, wV))
                                    + self.lmbda*U_full, wHalf.T
                                   )
            )
            M = np.eye(H.shape[0]) - H
            M_blocks = [
                    M[i*(classes-1) : (i+1)*(classes-1), 
                      i*(classes-1) : (i+1)*(classes-1)] for i in range(n)
                    ]
            H_blocks = [
                    H[i*(classes-1) : (i+1)*(classes-1),
                      i*(classes-1) : (i+1)*(classes-1)] for i in range(n)
                    ]
            xi = np.sqrt((1-p)/p)
            cooks = [
                    np.matmul(np.matmul(xi[i,:,None].T,
                        np.matmul(np.matmul(np.linalg.inv(M_blocks[i]),
                            H_blocks[i]), np.linalg.inv(M_blocks[i]))), 
                        xi[i,:,None])[0][0] for i in range(n)
            ]

        else:
            U = np.vstack((np.zeros((views, n*views + views)),
                           np.hstack(( np.zeros((n*views, views)), self.K ))
                           ))
            w_mat = np.kron(self.w.T, np.eye(n))
            wV = np.matmul(w_mat, self._V)

            eta = np.matmul( wV, alpha )

            p, Wdiag, W_diff = self._link_fun(eta, 0, self.model, 'alpha')

            WBV = np.sqrt(Wdiag)*wV 

            HatMat = np.matmul(
                    WBV, np.linalg.solve(np.matmul(Wdiag.T*wV.T, wV) 
                                         + self.lmbda*U, WBV.T)
                    )

            resids = np.sqrt(Wdiag)*W_diff

            cooks = (resids**2).T*(np.diag(HatMat)/(1-np.diag(HatMat))) 

        return cooks
