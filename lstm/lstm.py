import numpy as np
from scipy.special import expit

#WARNING there is currently an error somewhere where something becomes Nan
#not immediatley reproducible, suspect it has to do with how the x_list is initialized
#suspect it occurs if the numbers appearing in x_list are sufficently small, 
#i.e. arithmetic underflow, warrants further investigation

#some functions that will be used

#the derivative of the tanh function
def dtanh(x):
	ans = 1 - np.tanh(x)**2
	return ans

#derivative of the sigmoid
def dsigma(x):
	ans = expit(x)**2 * np.exp(-x)
	return ans

#the loss function ||y - \overline{y}||^{2}
#yy -> an array of predictions, where the 0th position gives the prediction
def loss(yy, y):
	ans = (y - yy[0])**2
	return ans
	
#the derivative of the loss function
def dloss(yy,y):
	d = np.zeros_like(yy)
	d[0] = 2*(y - yy[0])
	return d	

#initialize the paramaters as global variables, to be passed throughout the code
#could've been implemented better, but this is meant merely to learn the algorithm

#dimension -> the dimension of the input
#neurons -> number of neurons
nneurons = 500
inDim = 50
dim = inDim + nneurons

params = {}
#initialize the weights as random values
params['Wf'] = np.random.randn(nneurons, dim)
params['Wi'] = np.random.randn(nneurons, dim)
params['Wc'] = np.random.randn(nneurons, dim)
params['Wo'] = np.random.randn(nneurons, dim)

#initialize the biases as 0
params['bf'] = np.zeros(nneurons)
params['bi'] = np.zeros(nneurons)
params['bc'] = np.zeros(nneurons)
params['bo'] = np.zeros(nneurons)
params['Ct'] = np.zeros(nneurons)
params['ht'] = np.zeros(nneurons)

#initialize the derivatives as 0
params['Wc_diff'] = np.zeros_like(params['Wc'])
params['Wi_diff'] = np.zeros_like(params['Wi']) 
params['Wf_diff'] = np.zeros_like(params['Wf']) 
params['Wo_diff'] = np.zeros_like(params['Wo']) 
params['bi_diff'] = np.zeros_like(params['bi']) 
params['bf_diff'] = np.zeros_like(params['bf']) 
params['bo_diff'] = np.zeros_like(params['bo'])
params['bc_diff'] = np.zeros_like(params['bc'])

class LSTMNode():
	'''
	An implementation of a single node of a Long Short Term Memory network node
	'''
	
	def __init__(self):
		'''Initializes the network with h_{t} and C_{t} as all 0'''
		global nneurons
		self.ht = np.zeros(nneurons)
		self.Ct = np.zeros(nneurons)
	
	def forward(self, xx, ht_1 = None, Ct_1 = None):
		'''The forward pass of the system
			see: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
		
		params:
			xx -> the input to pass through this node
			ht_1 -> h_{t-1}, assumed to be all 0 if not specified
			Ct_{1} -> C_{t-1} assumed to be all 0 if not specified
		'''
		if ht_1 is None:
			ht_1 = np.zeros_like(self.ht)
		if Ct_1 is None:
			Ct_1 = np.zeros_like(self.Ct)
		x = np.concatenate([ht_1,xx])
		self.ft = expit(np.dot(params['Wf'], x) + params['bf'])
		self.it = expit(np.dot(params['Wi'], x) + params['bi'])
		self.Ctt = np.tanh(np.dot(params['Wc'], x) + params['bc'])
		self.Ct = self.ft * Ct_1 + self.it * self.Ctt
		self.ot = expit(np.dot(params['Wo'], x) + params['bo'])
		self.ht = self.ot*np.tanh(self.Ct)
		
		#backpropogation needs x, h_{t-1}, and C_{t-1}
		self.ht_1 = ht_1
		self.Ct_1 = Ct_1
		self.x = x

	def backward(self, dldh, dldc):
		'''Performs backpropogations throughout the node
			see: http://nicodjimenez.github.io/2014/08/08/lstm.html
		
		params:
			dldh -> the derivative of the loss function with respect to h at time t
			dldc -> the derivative of the loss function at time t+1 with respect to C
		'''
		global params
		global inDim
		
		#derivatives along the output
		dCt = self.ot * dldh + dldc
		do = self.Ct * dldh
		di = self.Ctt * dCt
		dCtt = self.it * dCt
		df = self.Ct * dCt
		
		#derivative along the calculations in this node
		di_calc = dsigma(self.it) * di 
		df_calc = dsigma(self.ft) * df 
		do_calc = dsigma(self.ot) * do 
		dCtt_calc = dtanh(self.Ctt) * dCtt
		
		#derivatives along the input
		params['Wi_diff'] += np.outer(di_calc, self.x)
		params['Wf_diff'] += np.outer(df_calc, self.x)
		params['Wo_diff'] += np.outer(do_calc, self.x)
		params['Wc_diff'] += np.outer(dCtt_calc, self.x)
		params['bi_diff'] += di_calc
		params['bf_diff'] += df_calc       
		params['bo_diff'] += do_calc
		params['bc_diff'] += dCtt_calc
		
		#A parameter we'll need later
		xx = np.zeros_like(self.x)
		xx += np.dot(params['Wi'].T, di_calc)
		xx += np.dot(params['Wf'].T, df_calc)
		xx += np.dot(params['Wo'].T, do_calc)
		xx += np.dot(params['Wc'].T, dCtt_calc)
		
		#the derivatives going out the the next (previous? next backwards)
		self.dCt_out = dCt * self.ft
		self.dht_out = xx[inDim:]
		
	
		
class lstm():

	def __init__(self):
		'''Initializes the network with no nodes and no inputs'''
		self.x_list = []
		self.node_list = []
	
	def add_input(self, x):
		'''Adds an input to the network
		
		params:
			x -> The input to be added
		'''
		global nneurons
		global dim
		self.x_list.append(x)
		#If there's more in the input list than the number of LSTM nodes to handle it
		#need a new node to deal with it
		if len(self.x_list) > len(self.node_list):
			self.node_list.append(LSTMNode())
		
		#If this is the first node, need a forward pass to generate everything, without previous entries
		if len(self.x_list) == 1:
			self.node_list[0].forward(x)
		
		#otherwise, we have history to work with
		else:
			C_prev = self.node_list[len(self.x_list) - 2].Ct
			h_prev = self.node_list[len(self.x_list) - 2].ht
			self.node_list[len(self.x_list) - 1].forward(x, C_prev, h_prev)
			
	def y_list_update(self, y_list):
		'''Performs the actual backpropogation throughout the network
		
		params:
			y_list -> A list of the outputs, assumes to be of the same length as the list of
						inputs to the nodes
		'''
		global nneurons
		
		#we're gonna work backwards through the graph, thus grab the last index
		j = len(self.x_list) - 1
		#1st node...
		losstot = loss(self.node_list[j].ht, y_list[j])
		diff_h = dloss(self.node_list[j].ht, y_list[j])
		diff_C = np.zeros(nneurons)
		self.node_list[j].backward(diff_h, diff_C)
		j -= 1
		
		#...and the rest
		while j >= 0:
			losstot += loss(self.node_list[j].ht, y_list[j])
			diff_h = dloss(self.node_list[j].ht, y_list[j])
			diff_C = self.node_list[j + 1].dCt_out
			diff_h += self.node_list[j + 1].dht_out
			self.node_list[j].backward(diff_h, diff_C)
			j -= 1 
		
		#and train the model
		self.__update()
		#output the loss so, that we can monitor that it's actually training
		return losstot
		
	def clear(self):
		'''A method to clear the list of inouts, for training purposes'''
		self.x_list = []
		
	def __update(self, mu = 1):
		'''An implementation of simple gradient descent.
		
		params:
			mu -> the learning rate
		'''
		global params
		params['Wc'] -= mu * params['Wc_diff']
		params['Wi'] -= mu * params['Wi_diff']
		params['Wf'] -= mu * params['Wf_diff']
		params['Wo'] -= mu * params['Wo_diff']
		params['bc'] -= mu * params['bc_diff']
		params['bi'] -= mu * params['bi_diff']
		params['bf'] -= mu * params['bf_diff']
		params['bo'] -= mu * params['bo_diff']
		params['Wc_diff'] = np.zeros_like(params['Wc'])
		params['Wi_diff'] = np.zeros_like(params['Wi']) 
		params['Wf_diff'] = np.zeros_like(params['Wf']) 
		params['Wo_diff'] = np.zeros_like(params['Wo']) 
		params['bc_diff'] = np.zeros_like(params['bc'])
		params['bi_diff'] = np.zeros_like(params['bi']) 
		params['bf_diff'] = np.zeros_like(params['bf']) 
		params['bo_diff'] = np.zeros_like(params['bo']) 
		

machine = lstm()		
y_list = [-0.5, 0.2, 0.1, -0.5]
x_list = [np.random.random(inDim) for _ in y_list]
print(x_list)

for _ in range(100):
	for x in x_list:
		machine.add_input(x)
	tloss = machine.y_list_update(y_list)
	print(tloss)
	machine.clear()