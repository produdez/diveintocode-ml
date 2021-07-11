import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#! init ...............................
class SimpleInitializer:
  """
  Simple initialization with Gaussian distribution
  Parameters
  ----------
  sigma : float
    Standard deviation of Gaussian distribution
  """
  def __init__(self, sigma):
    self.sigma = sigma
  def W(self, n_nodes1, n_nodes2):
    """
    Weight initialization
    Parameters
    ----------
    n_nodes1 : int
      Number of nodes in the previous layer
    n_nodes2 : int
      Number of nodes in the later layer
    Returns
    ----------
    W :
    """
    W = self.sigma * np.random.randn(n_nodes1,n_nodes2)
    return W
  def B(self, n_nodes2):
    """
    Bias initialization
    Parameters
    ----------
    n_nodes2 : int
      Number of nodes in the later layer
    Returns
    ----------
    B :
    """
    B = self.sigma * np.random.randn(1,n_nodes2)
    return B
#! activations..............

class ActivationFunction():
  def forward(self,A):
    pass
  def backward(self,dZ):
    pass
class Sigmoid(ActivationFunction):
  def func(self,A):
    return 1/(1+np.exp(-A))
  def forward(self,A):
    self.A = A
    return self.func(A)
  def backward(self,dZ):
    A = self.A
    dA = dZ * (1 - self.func(A))@self.func(A)
class Tanh(ActivationFunction):
  def forward(self,A):
    self.A = A
    Z = (np.exp(A) - np.exp(-A)) / (np.exp(A) + np.exp(-A))
    return Z
  def backward(self,dZ):
    A = self.A
    dA = dZ * (1 - np.tanh(A) ** 2)
    return dA
  
class SoftMax(ActivationFunction):
  def forward(self,A):
    self.A = A
    Z = np.exp(A) / np.sum(np.exp(A), axis = 1).reshape(-1,1)
    return Z
  def backward(self,Z,Y):
    A = self.A
    nb = Z.shape[0]
    dA = 1/nb * (Z - Y)
    return dA

#! fc............................

class FC:
    """
    Number of nodes Fully connected layer from n_nodes1 to n_nodes2
    Parameters
    ----------
    n_nodes1 : int
        Number of nodes in the previous layer
    n_nodes2 : int
        Number of nodes in the later layer
    initializer: instance of initialization method
    optimizer: instance of optimization method
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        # Initialize
        self.optimizer = optimizer
        self.n_nodes1, self.n_nodes2 = n_nodes1, n_nodes2
        # Initialize self.W and self.B using the initializer method
        self.W = initializer.W(n_nodes1, n_nodes2)
        self.B = initializer.B(n_nodes2)
        pass
    def forward(self, X):
        """
        forward
        Parameters
        ----------
        X : The following forms of ndarray, shape (batch_size, n_nodes1)
            入力
        Returns
        ----------
        A : The following forms of ndarray, shape (batch_size, n_nodes2)
            output
        """        
        self.X = X
        A = X @ self.W + self.B
        return A
    def backward(self, dA):
        """
        Backward
        Parameters
        ----------
        dA : The following forms of ndarray, shape (batch_size, n_nodes2)
            Gradient flowing from behind
        Returns
        ----------
        dZ : The following forms of ndarray, shape (batch_size, n_nodes1)
            Gradient to flow forward
        """
        # update
        self.dA = dA

        self = self.optimizer.update(self)
        return self.dZ()
    def dB(self):
        dB = self.dA.sum(axis = 0).reshape(1,-1)  
        return dB
    def dW(self):
        dW = self.X.T @ self.dA
        return dW
    def dZ(self):
        dZ = self.dA @ self.W.T
        return dZ



#! mini batch...........................
class GetMiniBatch:
    """
Iterator to get a mini-batch
    Parameters
    ----------
    X : The following forms of ndarray, shape (n_samples, n_features)
      Training data
    y : The following form of ndarray, shape (n_samples, 1)
      Correct answer value
    batch_size : int
      Batch size
    seed : int
      NumPy random number seed
    """
    def __init__(self, X, y, batch_size = 20, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self._X = X[shuffle_index]
        self._y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)
    def __len__(self):
        return self._stop
    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self._X[p0:p1], self._y[p0:p1]        
    def __iter__(self):
        self._counter = 0
        return self
    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self._X[p0:p1], self._y[p0:p1]

#! SGD................................................................
class SGD:
    """
    Stochastic gradient descent
    Parameters
    ----------
    lr : Learning rate
    """
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, layer):
        """
        Update weights and biases for a layer
        Parameters
        ----------
        layer : Instance of the layer before update
        """
        #update
        layer.B += - self.lr * layer.dB()
        layer.W += - self.lr * layer.dW()
        return layer

class FinishedDeepNeuralNetworkClassifier():
    def __init__(self,encoder, max_iter = 5, lr = 0.1, 
               layers_n_nodes = [400,200,10],
               optimizer_class = SGD,
               activation_class = Tanh, initializer_class = SimpleInitializer,
               verbose = False, debug = False):
        self.epoch = max_iter
        self.verbose = verbose
        self.debug = debug
        self.lr = lr
        #other non-parametric vars:
        self.encoder = encoder
        self.sigma = 0.01
        self.batch_size = 20 # batch size 
 
        #prep layers
        self.layers_n_nodes = layers_n_nodes
        self.initializer_class = initializer_class
        self.activation_class = activation_class
        self.optimizer_class = optimizer_class
        self.layers = []
        self.activations = []
        for i in range(len(layers_n_nodes)): 
          if i == 0: continue #specify first layer later when have X
          n_nodes1 = layers_n_nodes[i-1]
          n_nodes2 = layers_n_nodes[i]
          layer = FC(n_nodes1, n_nodes2, initializer_class(self.sigma), optimizer_class(self.lr))
          self.layers.append(layer)
          if i != len(layers_n_nodes)- 1: #last activation is softmax!
            self.activations.append(activation_class())
          else:
            self.activations.append(SoftMax())
        
    def enum_layer_act(self, rev = False):
        zipped = zip(self.layers, self.activations)
        if rev:
          return enumerate(reversed(list(zipped)))
        return enumerate(zipped)
    def forward_prop(self,X):
        Z = X
        for i, (layer, activation) in self.enum_layer_act():
          A = layer.forward(Z)
          Z = activation.forward(A)
          if self.debug:
            print(f'Z{i+1}: ', Z.shape, A.shape)
        return Z

    def backward_prop(self,Z,y):
        dA = self.activations[-1].backward(Z,y)
        for i, (layer, activation) in self.enum_layer_act(rev = True):
          if i == 0: #last layer has different activation backward!
            dZ = layer.backward(dA)
            continue
          dA = activation.backward(dZ)
          dZ = layer.backward(dA)
        
    
    def cross_entropy_error(self,Z,y):
        return (np.log(Z) * y).sum() / (- len(Z))

    def predict(self,X):
        y = np.zeros(X.shape[0])
        Z  = self.forward_prop(X)
        return self.encoder.transform(np.argmax(Z, axis = 1).reshape(-1,1))


    def fit(self,X,y, X_val = None, y_val = None):
        first_layer = FC(X.shape[1], self.layers_n_nodes[0], self.initializer_class(self.sigma), self.optimizer_class(self.lr))
        self.layers.insert(0,first_layer)
        self.activations.insert(0,self.activation_class())
        #prepare
        self.n_features = X.shape[1]
        self.lenx = len(X)
        self.batch_count = len(GetMiniBatch(X,y,batch_size= self.batch_size)) #for debug

        if self.verbose:
            print('X shape: ', X.shape, 'type: ', X.dtype)
            print('Batch count: ', self.batch_count)
            for i, (layer, activation) in self.enum_layer_act():
              print(f'Layer {i+1}: ', layer.n_nodes1, layer.n_nodes2)
              print(f'Activ: {i+1}:', activation.__class__.__name__)

        #train
        self.loss = np.zeros(self.epoch)
        self.accuracy = np.zeros(self.epoch)
        for i in range(self.epoch): #one full data ilteration
            if self.verbose: print('Epoch: ', i)
            self.get_mini_batch = GetMiniBatch(X, y, batch_size=self.batch_size)
            for idx, (mini_X_train, mini_y_train) in enumerate(self.get_mini_batch):
                if self.debug: print('Current batch: ', idx)
                #train mini_batch
                Z = self.forward_prop(mini_X_train)
                self.backward_prop(Z,mini_y_train)


            #record loss data
            Z = self.forward_prop(X)
            self.loss[i] = self.cross_entropy_error(Z,y)
            train_pred = self.predict(X)
            self.accuracy[i]  = accuracy_score(train_pred,y)
            if self.verbose:
                print(f'Loss {i}:', self.loss[i])
                print(f'Acc {i}:', self.accuracy[i])
                
        #verbose
        if self.verbose:
            print('Final train loss:',self.loss[-1])
            print('Final train accuracy:',self.accuracy[-1])

# #data set
# from keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# #reshape
# X_train = X_train.reshape(-1, 784)
# X_test = X_test.reshape(-1, 784)
# #scaling
# X_train = X_train.astype(np.float)
# X_test = X_test.astype(np.float)
# X_train /= 255
# X_test /= 255
# #one hot encode for multiclass labels!
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
# y_train_one_hot = enc.fit_transform(y_train[:, np.newaxis])
# y_test_one_hot = enc.transform(y_test[:, np.newaxis])
# #validation split
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_one_hot, test_size=0.2)

# model = FinishedDeepNeuralNetworkClassifier(
#     enc,
#     verbose= True
# )
# model.fit(X_train,y_train)