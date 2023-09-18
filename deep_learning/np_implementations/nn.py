import numpy as np
from sklearn.metrics import mean_squared_error
from np_implementations.optimiser import Optimiser

# TODO: factor the learning rate
class BaseNN():
    def __init__(self,lr = 0.01, batch_size = 512,beta=0.01, beta_v=0.09, beta_s=0.999,eps=1e-8,tol=0.01,max_iter = 100,optimiser="adam"):

        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.beta = beta
        self.beta_v = beta_v
        self.beta_s = beta_s
        self.eps = eps
        self.optimiser = optimiser
        self.output = None
        
    def forward(self,A):
        pass
    
    def backward(self,grad_A):
        pass

    def fit(self, X: np.ndarray = None ,y : np.ndarray = None):
        
        X = X.T

        m = X.shape[1]

        _iter = 0
        
        
        optimiser = self.initialiseOptimiser(X,y)
        
        while _iter < self.max_iter: 
            
            optimiser.run()
            
            

            cost = mean_squared_error(y,self.forward(X).ravel())
            if cost < self.tol:
                print(cost)
                break
        
            _iter += 1

        print(optimiser.tracker)
            
        print(self)
        
        
    def initialiseOptimiser(self,X,y):
        return Optimiser(self,self.batch_size,X,y)

    def predict(self,X):
        
        X = X.T
        A = self.forward(X)

        pred = np.where(A>0.5,1,0)

        return pred.ravel()

