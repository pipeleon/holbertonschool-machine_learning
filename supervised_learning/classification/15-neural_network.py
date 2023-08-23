#!/usr/bin/env python3
"""Task 8 Classification"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """Single neuron performing binary classification"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-z1))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        L = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return L

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.where(A < 0.5, 0, 1)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.__b2 -= alpha * db2
        self.__W2 -= alpha * dw2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network"""
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
            if graph:
                x = np.arange(0, iterations + 1, step)
                size = iterations // step + 1
                if iterations % step:
                    size += 1
                    np.append(x, iterations)
                y = np.empty((size,))

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)

            cost = self.cost(Y, A2)
            if (verbose or graph) and i % step == 0:                
                if verbose:
                    print('Cost after ' + str(i) + ' iterations: ' + str(cost))
                if graph:
                    y[i // step] = cost

            self.gradient_descent(X, Y, A1, A2, alpha)

        if verbose:
            print('Cost after ' + str(iterations) + ' iterations: ' + str(cost))
        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.suptitle("Training Cost")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.savefig("15-cost-train.png")

        return self.evaluate(X, Y)
