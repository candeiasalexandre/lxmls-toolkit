import sys
sys.path.append("/home/candeiasalexandre/code/lxmls-toolkit/lxmls/readers")
import numpy as np
import galton as galton
import matplotlib.pyplot as plt
import ipdb

def error_linear(x_data, y_data, w):
    """
        x_data - n_samples x n_features
        y_data - n_samples x output
        w - n_features x 1
    """
    error = np.matmul(x_data, w) - y_data
    #ipdb.set_trace()
    return error

def get_linear_grad(x_data, y_data, w):
    """
        x_data - n_samples x n_features
        y_data - n_samples x output
        w - n_features x 1
    """
    gradient = np.matmul(x_data.transpose(), 2.0*error_linear(x_data, y_data, w)) / y_data.shape[0]
    #ipdb.set_trace()
    return gradient

def linear_gradient_descent(x_data, y_data, max_iter=1000, min_tol=0.001, descent_step=0.0001):
    """
        x_data - n_samples x n_features
        y_data - n_samples x output
    """
    
    w = np.zeros((x_data.shape[1],1))
    w_old = w + 0.0
    for k in range(max_iter):
        w = w_old - descent_step * get_linear_grad(x_data, y_data, w_old)
        update = np.linalg.norm(w-w_old)
        if(update <= descent_step):
            break
        #ipdb.set_trace()
        w_old = w
    return w, k


galton_data = galton.load()
#without bias
x_data = galton_data[:,0]
x_data = x_data.reshape(x_data.shape[0], 1)
y_data = galton_data[:,1]
y_data = y_data.reshape(y_data.shape[0], 1)
w_fit, k = linear_gradient_descent(x_data, y_data)
print("Parameter: {}, Iteration: {}".format(w_fit, k))

    