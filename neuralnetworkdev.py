import numpy as np
import scipy as sc
from pathlib import Path
from numba import jit,cuda

def ActivationFunctionValidation(act_f):
    if type(act_f) == str:
        if act_f == "linear" : 
            _f = lambda x : x
            _df = lambda x : np.ones(np.shape(x))
        elif act_f == "relu" : 
            _f = lambda x : x * (x>0)
            _df = lambda x : 1 * (x>0)
        elif act_f == "sigm" : 
            _f = lambda x : 1.0/(1.0 + np.exp(-x))
            _df = lambda x :  1.0/(1.0 + np.exp(-x)) * (1 - 1.0/(1.0 + np.exp(-x)))
        elif act_f == "tanh" :
            _f = lambda x : np.tanh(x)
            _df = lambda x : 1.0-np.tanh(x)**2
        else: 
            raise NameError(f"The function '{act_f}' is not recognized. The function needs to be one of the integrated activation functions ('linear', 'relu', 'sigm', 'tanh').")
    elif type(act_f[0]) == type(lambda x:0):
        _f = act_f[0]
        if not (type(_f(np.random.rand(5))) == np.ndarray and type(_f(5)) in [int,float,np.int_,np.float_] and _f(np.random.rand(5)).shape == (5,)):
            raise AttributeError("act_f must be an algebraic function.")
    elif type(act_f[1]) == type(lambda x:0):
        _df = act_f[1]
        if not (type(_df(np.random.rand(5))) == np.ndarray and type(_df(5)) in [int,float,np.int_,np.float_] and _df(np.random.rand(5)).shape == (5,)):
            raise AttributeError("The derivative of act_f must be an algebraic function.")    
    else:
        raise TypeError("act_f needs to give two functions ( the activation function and its derivative) or a string with the name of one of the integrated activation functions ('linear', 'relu', 'sigm', 'tanh').")
    return _f,_df

def loss(cost_f:str, predicted_output:np.ndarray, real_output:np.ndarray, returnDerivativeCost = False):
    if predicted_output.shape != real_output.shape:
        raise IndexError("The ouput shapes don't match.")
    if cost_f == "MSE":
        _f = lambda p_out,r_out : np.mean((p_out-r_out)**2,axis=1*(np.ndim(p_out)>1),keepdims=True)
        _df = lambda p_out,r_out : p_out - r_out
    else:
        raise NameError(f"The function '{cost_f}' is not recognized. The function needs to be one of the integrated cost functions ('MSE').")
    if returnDerivativeCost:
        return _df(predicted_output, real_output)
    return _f(predicted_output, real_output)


class Neuron:

    def __init__(self, n_input:int, act_f):
        self.__act_f,self.__dact_f = ActivationFunctionValidation(act_f)
        self.__weights = np.random.rand(n_input)*2-1
        self.__bias = np.random.rand()*2-1
        
    def apply(self, input:np.ndarray, returnWeightedSum=False) -> np.float_|tuple:
        if input.ndim != 1:
            raise TypeError("The input must to be a vector.")
        elif len(self.__weights) != input.shape[0]:
            raise IndexError(f"The neuron only takes {len(self.__weights)} while {len(input)} were given.")
        _weighted_sum = np.dot(self.__weights,input) + self.__bias
        if returnWeightedSum:
            return _weighted_sum,self.__act_f(_weighted_sum)
        return self.__act_f(_weighted_sum)
    
    def __len__(self):
        return len(self.__weights)
    
    @property
    def weights(self):
        return self.__weights
    
    @weights.setter
    def weights(self, new_weights:np.ndarray):
        if new_weights.ndim != 1:
            raise IndexError("'weights has to be a vector")
        self.__weights = new_weights

    @property
    def bias(self):
        return self.__bias
    
    @bias.setter
    def bias(self, new_bias:np.float_):
        if not isinstance(new_bias,np.float_):
            raise TypeError(f"'bias' must be a float")
        self.__bias = new_bias

    @property
    def act_f(self):
        return self.__act_f
    
    @property
    def dact_f(self):
        return self.__dact_f


class Layer:

    def __init__(self, n_layer=0, n_prev_layer=0):
        self.__size = n_layer
        self.__input_size = n_prev_layer

    def __len__(self):
        return self.__size
    
    @property
    def size(self):
        return self.__size
    
    @property
    def input_size(self):
        return self.__input_size


class FullyConnectedLayer(Layer):
    
    def __init__(self, n_layer:int, input_size:int,act_f):
        self.__act_f,self.__dact_f = ActivationFunctionValidation(act_f)        
        super().__init__(n_layer,input_size)
        self.__neural_layer = [Neuron(self.input_size,act_f) for _ in range(self.size)]
        self.__weights = np.array([ neuron.weights for neuron in self.__neural_layer ])
        self.__bias = np.array([ neuron.bias for neuron in self.__neural_layer ])[:,np.newaxis]

    def apply(self, input:np.ndarray, returnWeightedSums=False) -> np.ndarray:
        if input.ndim > 2:
            raise TypeError("The input must be a vector or a matrix.")
        elif input.ndim == 1 and self.input_size != input.shape[0]:
            raise IndexError(f"The layer only takes {self.input_size} arguments while {input.shape[0]} were given.")
        elif input.ndim == 2 and self.input_size != input.shape[1]:
            raise IndexError(f"The layer only takes {self.input_size} arguments while {input.shape[1]} were given.")
        if input.ndim == 1:
            _input_vectors = input[:,np.newaxis]
        else:
            _input_vectors = input.T
        if returnWeightedSums:
            return (self.__weights.dot(_input_vectors) + self.bias).T
        return (self.__act_f(self.__weights.dot(_input_vectors) +  self.bias)).T
    
    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, new_weights:np.ndarray):
        if new_weights.shape != (self.size,self.input_size):
            raise IndexError(f"'weights' must be a matrix with shape ({self.size}, {self.input_size})")
        for i,neuron in enumerate(self.__neural_layer):
            neuron.weights = new_weights[i,:]
        self.__weights = new_weights
        

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, new_bias:np.ndarray):
        if new_bias.shape != (self.size, 1):
            raise IndexError(f"'bias' must be a vector with size {self.size}")
        for i,neuron in enumerate(self.__neural_layer):
            neuron.bias = new_bias[i,0]
        self.__bias = new_bias
        

    @property
    def act_f(self):
        return self.__act_f
    
    @property
    def dact_f(self):
        return self.__dact_f


class Model:

    def __init__(self):
        self.__input_size = 0
        self.__topology = []
        self.__layers = []

    def __len__(self):
        return len(self.__layers)
    
    def add(self,layer:Layer):
        if np.shape(self.__layers) == (0,):
            self.__input_size = layer.input_size
        self.__topology.append(layer.size)
        self.__layers.append(layer)

    def pop(self):
        if np.shape(self.__layers) != (0,):
            del self.__topology[-1]
            del self.__layers[-1]
    
    def input_error(self, input:np.ndarray):
        if np.shape(self.__layers) == (0,):
            print("The model is empty.")
        elif input.ndim > 2 or input.ndim < 2:
            raise TypeError("The input must be a vector or a matrix (np.ndim(input) = 2).")
        elif self.__input_size != input.shape[1]:
            raise IndexError(f"The model only takes {self.__input_size} arguments while {input.shape[1]} were given.")
        
    def output_error(self, output:np.ndarray):
        if np.shape(self.__layers) == (0,):
            print("The model is empty.")
        elif output.ndim > 2 or output.ndim < 2:
            raise TypeError("'output' must either be a vector or a matrix (np.ndim(output) = 2) where each line is a valid output for the model.")
        elif self.__layers[-1].size != output.shape[1]:
            raise IndexError(f"The model expects an output with {self.__layers[-1].size} arguments while {output.shape[1]} were given.")

    @property
    def input_size(self):
        return self.__input_size
    
    @property
    def topology(self):
        return self.__topology
    
    @property
    def layers(self):
        return self.__layers


class FullyConnected1DModel(Model):
           
    def __init__(self, input_size=0, topology=None,act_f=None):
        super().__init__()
        if (np.shape(topology) == (0,) and np.shape(act_f) == (0,)) or (np.shape(topology) == ()):
            return None
        if np.shape(topology) != np.shape(act_f) and np.shape(act_f) != ():
            raise IndexError("'act_f' must be a string or a list with the same size as 'topology'.")
        elif len(np.shape(topology)) != 1:
            raise TypeError("'topology' must be a vector.")
        if isinstance(act_f,str):
            _act_f=[act_f for _ in range(len(topology))]
        else:
            _act_f = act_f
        _layer_sizes = np.concatenate(([input_size],topology))
        for i_layer in range(len(_layer_sizes)-1):
            super().add(FullyConnectedLayer(n_layer=_layer_sizes[i_layer+1],input_size=_layer_sizes[i_layer],act_f=_act_f[i_layer]))

    def add(self, layer:FullyConnectedLayer):
        if not isinstance(layer,FullyConnectedLayer):
            raise TypeError("'FullyConnected1DModel' can only add 'FullyConnectedLayer'")
        elif np.shape(self.layers) != (0,) and layer.input_size != self.layers[-1].size:
            raise IndexError(f"The new layer must receive an input with size {self.layers[-1].size}, while the input size given was {layer.input_size}.")        
        super().add(layer)

    def apply(self, input:np.ndarray):
        _output = input
        for layer in self.layers:
            _output = layer.apply(_output)
        return _output
    
    def loss(self, cost_f:str, input:np.ndarray, real_output:np.ndarray):
        self.output_error(real_output)
        self.input_error(input)
        _output = self.apply(input)
        return loss(cost_f, _output, real_output)

    def train(self, input:np.ndarray, real_output:np.ndarray, cost_f:str, learning_rate=0.5):
        self.input_error(input)
        self.output_error(real_output)        
        #Forward Pass
        _layers_output = [input]
        for layer in self.layers:
            _layers_output.append(layer.apply(_layers_output[-1]))
        #Backward Pass
        _deltas = []        
        for i_layer in reversed(range(len(self))):
            if i_layer == len(self)-1:
                _i_deltas = (loss(cost_f, _layers_output[i_layer+1], real_output, returnDerivativeCost=True) * self.layers[i_layer].dact_f(_layers_output[i_layer+1])).T
            else:
                _i_deltas = (np.array(_deltas[0]).T.dot(self.layers[i_layer+1].weights)).T * self.layers[i_layer].dact_f(_layers_output[i_layer+1]).T
            _deltas.insert(0,_i_deltas)
            #Gradient Descent
            self.layers[i_layer].weights -= _i_deltas.dot(_layers_output[i_layer]) * learning_rate
            self.layers[i_layer].bias -= np.mean(_i_deltas,axis=1,keepdims=True) * learning_rate
        return np.mean(loss(cost_f, _layers_output[-1], real_output))

    def compile(self, input:np.ndarray, real_output:np.ndarray, cost_f:str, steps:int, learning_rate=0.5, show_on_step = 1):
        _losses = []
        for step in range(1,steps+1):
            _loss = self.train(input, real_output, cost_f, learning_rate)
            _losses.append(_loss)
            if step%show_on_step == 0:
                print(f"Step {step}/{steps} -> loss : {_loss}")
        return _losses
    
    def save(self,path:Path):
        _array_weights = [layer.weights for layer in self.layers]



