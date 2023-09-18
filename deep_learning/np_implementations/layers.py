import numpy as np
from np_implementations.nn import BaseNN


class Linear(BaseNN):
    def __init__(self,node_size,input_dims=None,lr=None,init_lr = 0.2,norm='l2',lambd=0.02,lr_decay_rate = 0.02,lr_decay=False,weight_init="he"):
        super().__init__()
        
        self.node_size = node_size
        self.weights = None
        self.bias = None
        self.Z = None
        self.prev_A = None
        
        self.lambd = lambd
        self.norm = norm
        
        self.weight_init = weight_init
        
        self.epoch_num = 0
        self.V_w = 0
        self.V_b = 0
        self.S_w = 0
        self.S_b = 0
        
        
        
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.lr_decay_rate = lr_decay_rate
        
        if lr != None:
            self.lr = lr
            self.init_lr = lr
            
        #input dims: n[l-1]
        if input_dims:
            self.init_params(input_dims)

        self.grads = {}
                    
    def init_params(self, input_dims):
        if self.weight_init == "he":
            scaler = np.sqrt(2/input_dims)
        elif self.weight_init == "xavier":
            scaler = np.sqrt(1/input_dims)
        
        self.weights = np.random.randn(self.node_size, input_dims) * scaler
        
        self.bias = np.zeros((self.node_size,1))
        
        #batch norm
        self.norm_beta = np.random.randn(self.node_size, input_dims) * scaler
        self.norm_gamma = np.zeros((self.node_size,1))

        
    def update(self):
        if self.norm == "l2":
            reg = self.lambd/self.grads['W'].shape[1] * np.linalg.norm(self.weights)
        
        elif self.norm == "l1":
            reg = self.lambd/self.grads['W'].shape[1] * np.linalg.norm(self.weights, ord=1)
            
        else:
            reg = 0
            
            
        if self.optimiser == "GDMomentum":
            self.V_w = self.beta*self.V_w + (1-self.beta)*self.grads['W']
            self.V_b = self.beta*self.V_b + (1-self.beta)*self.grads['b']
            
            self.weights = self.weights - self.lr * self.V_w + reg
            self.bias = self.bias - self.lr * self.V_w
            
            
        elif self.optimiser == "RMSprop":
            self.S_w = self.beta*self.V_w + (1-self.beta)*self.grads['W']**2
            self.S_b = self.beta*self.V_b + (1-self.beta)*self.grads['b']**2
            
            
            self.weights = self.weights - self.lr * self.grads['W']/np.sqrt(self.S_w+self.eps) + reg
            self.bias = self.bias - self.lr * self.grads['b']/np.sqrt(self.S_b+self.eps)
            
        elif self.optimiser == "adam":
            self.V_w = self.beta_v*self.V_w + (1-self.beta_v)*self.grads['W']
            self.V_b = self.beta_v*self.V_b + (1-self.beta_v)*self.grads['b']
            
            self.S_w = self.beta_s*self.S_w + (1-self.beta_s)*self.grads['W']**2
            self.S_b = self.beta_s*self.S_b + (1-self.beta_s)*self.grads['b']**2
            
            V_w_corr = self.V_w/(1-self.beta_v)
            V_b_corr = self.V_b/(1-self.beta_v)
            
            S_w_corr = self.S_w/(1-self.beta_s)
            S_b_corr = self.S_b/(1-self.beta_s)
            
            self.weights = self.weights - self.lr * V_w_corr/np.sqrt(S_w_corr+self.eps) + reg
            self.bias = self.bias - self.lr * V_b_corr/np.sqrt(S_b_corr+self.eps)
            
            self.norm_gamma = self.norm_gamma - self.lr * V_w_corr/np.sqrt(S_w_corr+self.eps) + reg
            self.norm_beta = self.norm_beta - self.lr * V_w_corr/np.sqrt(S_w_corr+self.eps) + reg
        
        #learning rate decay
        if self.lr_decay:
            self.lr = 1/(1+self.lr_decay_rate*self.epoch_num)*self.init_lr
        else:
            pass

        
        # self.weights = self.weights - self.lr * self.grads['W'] + reg
        
        # self.bias = self.bias - self.lr * self.grads['b']
        
        # #maintain numerical stablity
        # self.weights += 0.0001

    def forward(self, prev_A):
        # input: (input_dims, batch_size)

        if self.weights is None:
            self.init_params(prev_A.shape[0])

        # linear function (Z = Wx + b)
        # takes in A of previous layer, shape: (input_dims, batch_size)
        # outputs Z of shape: (node_size, batch_size)
        # print(self.weights)
        self.Z = np.dot(self.weights, prev_A) + self.bias

        # cache for backpropagation
        self.prev_A = prev_A
        
        # batch norm
        
        m = self.Z.shape[1]
            
        mean = (1/m) * np.sum(self.Z, axis=0)
        var = 1/(m)*np.sum((self.Z - mean)**2,axis=0)
        Z_norm = (self.Z - mean) / np.sqrt(var + self.eps)
        self.Z = self.norm_gamma*Z_norm + self.norm_beta
        
        return self.Z
    

    def backward(self, grad_Z):

        # grad_W, shape: (node_size, input_dims)
        grad_W = 1/self.prev_A.shape[1] * (np.dot(grad_Z, self.prev_A.T))

        # grad_b, shape: (node_size, 1)
        grad_b = 1/self.prev_A.shape[1] * (np.sum(grad_Z, axis=1, keepdims=True))

        # grad_prev_A, shape: (input_dims, batch_size)
        grad_prev_A = np.dot(self.weights.T, grad_Z)

        self.grads = {'W': grad_W, 'b': grad_b, 'prev_A': grad_prev_A,'Z': grad_Z}
        
        self.epoch_num += 1

        self.update()
        
        return grad_prev_A


class BaseAct():
    def __init__(self):
        self.A = None
        self.Z = None
    

class Sigmoid(BaseAct):
    
    #cache/return A
    def forward(self, Z):
                    
        self.A = 1 / (1 + np.exp(-Z))
        
        return self.A
    
    #return grad_Z
    def backward(self, grad_A):
        
        # grad_A: gradient of loss function wrt A of this layer
        # grad_Z, shape: (node_size, batch_size)
        grad_Z = grad_A * (self.A * (1 - self.A))
        
        return grad_Z
    
class ReLu(BaseAct):
    
    def forward(self, Z):
        
        #cache Z
        self.Z = Z
        
        self.A = Z * (Z > 0)
        
        return self.A
    
    def backward(self, grad_A):
                    
        grad_Z = grad_A * (self.Z > 0)
        
        return grad_Z
    
class TanH(BaseAct):
    
    def forward(self, Z):
        
        #cache Z
        self.Z = Z
        
        self.A = Z * (Z > 0)
        
        return self.A
    
    def backward(self, grad_A):
                    
        grad_Z = grad_A * (1 - self.A * self.A)
        
        return grad_Z
    
class Softmax(BaseAct):
    
    def forward(self, Z):
        
        #cache Z
        self.Z = Z
        
        self.A = (1/ np.sum(Z, axis=1)) * np.exp(Z)
        
        return self.A
    
    def backward(self, grad_A):
                    
        grad_Z = self.A * (grad_A - np.sum(grad_A * self.A, axis=0))
        
        return grad_Z
    
    
    
class InvertedDropout():
    def __init__(self,keep_prob=0.8):
        self.keep_prob = keep_prob
        self.rand_init = None
        
    def forward(self,A):
        self.rand_init = np.random.rand(A.shape[0],A.shape[1]) < self.keep_prob
        A = np.multiply(A,self.rand_init)
        A /= self.keep_prob
        
        return A
    
    def backward(self,grad_A):
    
        grad_A = np.multiply(grad_A,self.rand_init)
        
        return grad_A


