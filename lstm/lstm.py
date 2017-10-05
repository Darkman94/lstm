import numpy as np
from scipy.special import expit

class LSTM():
	'''
	An implementation of a Long Short Term Memory network
	see: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
	'''
	
	'''
	dimension -> the dimension of the input
	neurons -> number of neurons
	'''
	def __init__(self, neurons, dimension):
		self.inDim = dimension
		self.nneurons = neurons
		self.dim = self.inDim + self.nneurons 
		self.Wf = np.random.randn(self.nneurons, self.dim)
		self.Wi = np.random.randn(self.nneurons, self.dim)
		self.Wc = np.random.randn(self.nneurons, self.dim)
		self.Wo = np.random.randn(self.nneurons, self.dim)
		self.bf = np.zeros(self.nneurons)
		self.bi = np.zeros(self.nneurons)
		self.bc = np.zeros(self.nneurons)
		self.bo = np.zeros(self.nneurons)
		self.Ct = np.zeros(self.nneurons)
		self.ht = np.zeros(self.nneurons)
	
	'''
	xx -> an np array theat gives the input vector
	'''
	def forward(self, xx):
		x = np.concatenate([self.ht,xx])
		ft = expit(np.dot(self.Wf, x) + self.bf)
		it = expit(np.dot(self.Wi, x) + self.bi)
		Ctt = np.tanh(np.dot(self.Wc, x) + self.bc)
		self.Ct = ft * self.Ct + it * Ctt
		ot = expit(np.dot(self.Wo, x) + self.bo)
		self.ht = ot*np.tanh(self.Ct)
		
input = np.array([1,2,3])
network = LSTM(10,3)
for _ in range(1000):
	network.forward(input)
print(network.Ct)
print(network.ht)