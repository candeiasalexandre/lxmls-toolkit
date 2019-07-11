import ipdb

def greet(hour):
    #ipdb.set_trace()
    if hour < 12:
        print('Good morning!')
    elif hour >= 12 and hour < 20:
        print('Good afternoon!')
    else:
        print('Good evening!')

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X = np.linspace(-4, 4, 1000)
    plt.plot(X, X**2*np.cos(X**2))
    plt.show()
    #plt.savefig("simple.pdf")
    #greet(10)