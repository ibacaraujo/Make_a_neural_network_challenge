import numpy as np

class NeuralNetwork():
	def __init__(self):
		np.random.seed(1)
		self.synaptic_weights_0 = 2 * np.random.random((3, 4)) - 1
		self.synaptic_weights_1 = 2 * np.random.random((4, 1)) - 1
		
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
		
	def __sigmoid_derivative(self, x):
		return x * (1 - x)
		
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			inputs = training_set_inputs # 4 x 3
			hidden, output = self.think(inputs)
			#hidden = self.think(inputs, self.synaptic_weigts_0) # 4 x 4 
			#output = self.think(hidden, self.synaptic_weights_1) # 4 x 1
			output_error = training_set_outputs - output
			output_adjustment = np.dot(hidden.T, output_error * self.__sigmoid_derivative(output)) # 4 x 1
			hidden_error = np.dot(output_adjustment, self.synaptic_weights_1.T) # 4 x 4
			hidden_adjustment = np.dot(inputs.T, hidden_error * self.__sigmoid_derivative(hidden)) # 3 x 4
			self.synaptic_weights_1 += output_adjustment
			self.synaptic_weights_0 += hidden_adjustment
			#output = self.think(training_set_inputs)
			#error = training_set_outputs - output
			#adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			#self.synaptic_weights += adjustment
			
	def think(self, inputs):
		hidden = self.__sigmoid(np.dot(inputs, self.synaptic_weights_0))
		output = self.__sigmoid(np.dot(hidden, self.synaptic_weights_1))
		return hidden, output
		
if __name__ == "__main__":
	
	neural_network = NeuralNetwork()
	
	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights_0
	print neural_network.synaptic_weights_1
	
	# The training set. We have 4 examples, each consisting of 3 input values
	# and 1 output value
	training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = np.array([[0, 1, 1, 0]]).T
	
	# Train the neural network using a training set.
	# Do it 10,000 times and make small adjustments each time.
	neural_network.train(training_set_inputs, training_set_outputs, 10000)
	
	print "New synaptic weights after training: "
	print neural_network.synaptic_weights_0
	print neural_network.synaptic_weights_1
	
	# Test the neural network in a new situation.
	
	print "Considering the new situation [1, 0, 0] -> ?: "
	print neural_network.think(np.array([1, 0, 0]))[1]
