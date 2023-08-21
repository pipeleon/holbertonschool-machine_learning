#!/usr/bin/env python3
"""Task 7 Classification"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """Single neuron performing binary classification"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        L = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return L

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.where(A < 0.5, 0, 1)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y

        dw = (1 / m) * np.matmul(X, dz.T)
        new_W = self.__W - alpha * dw.T
        self.__W = new_W

        db = (1 / m) * np.sum(dz)
        new_b = self.__b - alpha * db
        self.__b = new_b

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_by_i = []
        for i in range(iterations):
            A = self.forward_prop(X)
            cost_by_i.append(self.cost(Y, A))
            if verbose and i % step == 0:
                print('Cost after ' + str(i) + ' iterations: ' + str(cost_by_i[i]))
            self.gradient_descent(X, Y, A, alpha)
        
        
        if graph:
            plt.plot(np.array(cost_by_i))
            plt.suptitle("Training Cost")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.savefig("costvi.png")

        return self.evaluate(X, Y)
