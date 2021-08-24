
#! ..... Convolution 1D

class Conv1DBatch:
  def __init__(self, filter_size = 3, n_input = 2, n_output = 3, optimizer = SGD(), initializer = ConvInitializer(), padding = 0):
    # Initialize
    self.optimizer = optimizer
    self.filter_size = filter_size
    # Initialize self.W and self.B using the initializer method
    self.filter_size = filter_size
    self.n_input = n_input
    self.n_output = n_output

    if initializer == None:
      self.W = np.ones((n_output, n_input,filter_size))
      self.B = np.ones(n_output)
    else:
      self.W = initializer.W(n_output, n_input, filter_size)
      self.B = initializer.B(n_output)
    #padding and stride
    self.padding = padding
        
  def forward(self,X): #! NOTE: X must be 3 dimensional (batch_size, channel_count, feature_count)
    self.batch_size = len(X)
    X = np.pad(X,[(0,0),(0,0),(self.padding, self.padding)], mode = 'constant')
    self.X = X
    result = []
    for x in self.X:
      result.append(self._forward_sample(x))
    # print('forward: ', np.array(result).shape)
    return np.array(result)
  def _forward_sample(self,X):    
    output = []
    for i in range(self.n_output):
      filt = self.W[i]
      bias = self.B[i]
      conv = self.convolve(filt, X)
      output.append(conv.sum(axis = 0) + bias)
    return np.array(output)

  def backward(self, dA):
    # update
    self.dA = dA
    self.dX = []

    for i in range(self.batch_size):
      self.dW = self.calc_dW(i)
      self.dB = self.calc_dB(i)
      self.dx = self.calc_dx(i)
      self = self.optimizer.update(self)
      self.dX.append(self.dx) #! keep updating while calculating error for prev layer
      # print('dx: ', i, self.dx.shape,self.dx)
    self.dX = np.array(self.dX)
    return self.dW, self.dB , self.dX
    

  def calc_dW(self,sample_index):
    dW = []
    dupped_dA = np.repeat(self.dA[sample_index][:,np.newaxis, : ], self.n_input, axis=1)
    for i in range(self.n_output): #convolve each output_channel through X
      conv = self.convolve(dupped_dA[i], self.X[sample_index])
      dW.append(conv) 
    return np.array(dW)
  def calc_dB(self,sample_index):
    return np.array(self.dA[sample_index].sum(axis = 1))

  def calc_dx(self, sample_index): #! careful
    pad_dA = self.pad_dA(sample_index) #match a with x
    flipped_W = np.flip(self.W,axis = 2).reshape(self.n_input,self.n_output, -1) # flip each filter and the in_out dim also
    output = []
    for i in range(self.n_input):
      filt = flipped_W[i]
      conv = self.convolve(filt, pad_dA)
      output.append(conv.sum(axis = 0))
    return np.array(output)
  def pad_dA(self,sample_index):
    array = self.dA[sample_index]
    n_features_in = array.shape[1]
    n_features_out = self.X.shape[-1]
    filter_size = self.filter_size
    padding = (n_features_out - 1 + filter_size - n_features_in) // 2
    return self.pad(array, padding)
  def pad(self,array,padding):
    return np.pad(array,[(0,0),(padding, padding)], mode = 'constant')
  def convolve(self,F,X):
    A = []
    filter_size = F.shape[-1]
    feature_count = X.shape[-1]
    n_out_features = (feature_count - filter_size) + 1
    for i in range(n_out_features):
      A.append((X[...,i : i + filter_size] * F).sum(axis=-1))
    return np.array(A).T
