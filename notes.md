Resources 
 - https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
 -  


### Nuron
```
import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 0
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994 
```


Variables: 
-----------------------------------------------
	[x1, x2, ...] -> Input nurons
	[y1, y2, ...] -> Output nurons
	[w1, w2, ...] -> weights
	b -> bias

-----------------------------------------------
	
Y = f(x1 * w1 + x2 * w2 + ... + b) where f is the activation function. 


E.G: 
-----------------------------------------------
	x = [2, 3]
	w = [1, 0]
	b = 4

	y = f(0*2 + 1*3 + 4) = f(7) = 0.999 // Using a sigmoid activation function. 


For level = 1: 
-----------------------------------------------
	Input1 -> Output1(Sol)


For level = n: 
-----------------------------------------------
	Input1 -> Output1, Input2 = Output1
	Input2 -> Output2, Input3 = Output2
	.
	.
	.
	InputN -> Output(Sol)



