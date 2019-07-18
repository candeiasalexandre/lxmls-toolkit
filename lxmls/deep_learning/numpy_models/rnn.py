import numpy as np
from lxmls.deep_learning.rnn import RNN
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyRNN(RNN):

    def __init__(self, **config):
        # This will initialize
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self.log_forward(input)[0])
        return np.argmax(p_y, axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in range(num_parameters):
            # Update weight
            self.parameters[m] -= learning_rate * gradients[m]

    def log_forward(self, input):

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        hidden_size = W_h.shape[0]
        nr_steps = input.shape[0]

        # Embedding layer
        z_e = W_e[input, :]

        # Recurrent layer
        h = np.zeros((nr_steps + 1, hidden_size))
        for t in range(nr_steps):

            # Linear
            z_t = W_x.dot(z_e[t, :]) + W_h.dot(h[t, :])

            # Non-linear
            h[t+1, :] = 1.0 / (1 + np.exp(-z_t))

        # Output layer
        y = h[1:, :].dot(W_y.T)

        # Softmax
        log_p_y = y - logsumexp(y, axis=1, keepdims=True)

        return log_p_y, y, h, z_e, input

    def backpropagation(self, input, output):

        '''
        Compute gradientes, with the back-propagation method
        inputs:
            x: vector with the (embedding) indicies of the words of a
                sentence
            outputs: vector with the indicies of the tags for each word of
                        the sentence outputs:
            gradient_parameters: vector with parameters gradientes
        '''

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        nr_steps = input.shape[0]

        log_p_y, y, h, z_e, x = self.log_forward(input)
        p_y = np.exp(log_p_y)

        # Initialize gradients with zero entrances
        gradient_W_e = np.zeros(W_e.shape)
        gradient_W_x = np.zeros(W_x.shape)
        gradient_W_h = np.zeros(W_h.shape)
        gradient_W_y = np.zeros(W_y.shape)

        # ----------
        # Solution to Exercise 1
        error_y = np.zeros((nr_steps, W_y.shape[0]))
        output_onehot = index2onehot(output, p_y.shape[1])
        for t in range(nr_steps):
            #import ipdb; ipdb.set_trace()
            error_y[t,:] = output_onehot[t, :] - p_y[t,:]

        error_r = np.zeros(W_y.shape[1])
        x_e = index2onehot(x, W_e.shape[0])
        for t in reversed(range(nr_steps)):
            #import ipdb; ipdb.set_trace()
            error_h = np.multiply( error_r + np.matmul(W_y.T, error_y[t,:]), np.multiply(h[t+1,:], 1.0-h[t+1,:]) )
            error_r = np.matmul(W_h.T, error_h)
            error_e = np.matmul(W_x.T, error_h)

            gradient_W_y += np.outer(error_y[t,:], h[t+1, :])
            gradient_W_h += np.outer(error_h, h[t-1, :])
            gradient_W_x += np.outer(error_h, z_e[t, :])
            gradient_W_e += np.outer(error_e, x_e[t,:]).T
           
            
        gradient_W_e = gradient_W_e * (-1.0/nr_steps)
        gradient_W_h = gradient_W_h * (-1.0/nr_steps)
        gradient_W_x = gradient_W_x * (-1.0/nr_steps)
        gradient_W_y = gradient_W_y * (-1.0/nr_steps)
        # End of Solution to Exercise 1
        # ----------

        # Normalize over sentence length
        gradient_parameters = [
            gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
        ]

        return gradient_parameters

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        nr_steps = input.shape[0]
        log_probability = self.log_forward(input)[0]
        return -log_probability[range(nr_steps), output].mean()
