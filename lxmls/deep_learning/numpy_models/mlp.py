import numpy as np
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot, logsumexp
import ipdb

class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_class_probabilities, _ = self.log_forward(input)
        return np.argmax(np.exp(log_class_probabilities), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """

        gradients = self.backpropagation(input, output)
        #ipdb.set_trace()

        learning_rate = self.config['learning_rate']
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):

            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]

            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]

    def log_forward(self, input):
        """Forward pass for sigmoid hidden layers and output softmax"""

        # Input
        tilde_z = input
        layer_inputs = []

        # Hidden layers
        num_hidden_layers = len(self.parameters) - 1
        for n in range(num_hidden_layers):

            # Store input to this layer (needed for backpropagation)
            layer_inputs.append(tilde_z)

            # Linear transformation
            weight, bias = self.parameters[n]
            z = np.dot(tilde_z, weight.T) + bias

            # Non-linear transformation (sigmoid)
            tilde_z = 1.0 / (1 + np.exp(-z))

        # Store input to this layer (needed for backpropagation)
        layer_inputs.append(tilde_z)

        # Output linear transformation
        weight, bias = self.parameters[num_hidden_layers]
        z = np.dot(tilde_z, weight.T) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

        return log_tilde_z, layer_inputs

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        num_examples = input.shape[0]
        log_probability, _ = self.log_forward(input)
        return -log_probability[range(num_examples), output].mean()

    def backpropagation(self, input, output):
        """Gradients for sigmoid hidden layers and output softmax"""

        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input)
        prob_y = np.exp(log_prob_y)

        num_examples, num_clases = prob_y.shape
        num_hidden_layers = len(self.parameters) - 1
        num_layers = self.num_layers

        # For each layer in reverse store the backpropagated error, then compute
        # the gradients from the errors and the layer inputs
        gradients = []
        errors = []

        
        # ----------
        # Solution to Exercise 2
        for n in reversed(range(num_layers)):
            W, b = self.parameters[n]
            if n == num_layers -1:
                last_error = log_prob_y - index2onehot(output, np.unique(output).shape[0])
            else:
                last_error = np.multiply(last_error, np.multiply(layer_inputs[n-1], 1.0-layer_inputs[n-1]))

            #W_gradient = (-1.0/num_examples) * np.sum(np.matmul(last_error[:, :, np.newaxis], layer_inputs[n-1][:, :, np.newaxis]), axis=2) + 0.0
            W_gradient = (-1.0/num_examples) * np.matmul(last_error.T, layer_inputs[n]) + 0.0
            b_gradient = (-1.0/num_examples) * np.sum(last_error, axis=0) + 0.0
            
            #ipdb.set_trace()
            last_error = np.matmul(W.T, last_error.T).T
            errors.append(last_error + 0.0)
            gradients.append((W_gradient, b_gradient))
        gradients.reverse()
        # End of solution to Exercise 2
        # ----------
        return gradients
