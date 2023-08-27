#!/usr/bin/env python3
"""Task 26 Classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork ():
    """Deep Neural Network performing binary classification"""
    def __init__(self, nx, layers, activation='sig'):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        tmp_w = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            ly = layers
            if i == 0:
                st = np.sqrt(2/nx)
                tmp_w['W' + str(i+1)] = np.random.normal(0, st, (ly[i], nx))
            else:
                st = np.sqrt(2/layers[i - 1])
                key = 'W' + str(i+1)
                tmp_w[key] = np.random.normal(0, st, (ly[i], ly[i-1]))
            tmp_w['b' + str(i+1)] = np.zeros((layers[i], 1))

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = tmp_w
        self.__activation = activation

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X

        for i in range(self.__L):
            if i == 0:
                z = np.matmul(self.__weights['W1'], X) + self.__weights['b1']
            else:
                W = self.__weights['W' + str(i+1)]
                A = self.__cache['A' + str(i)]
                z = np.matmul(W, A) + self.__weights['b' + str(i+1)]

            if i + 1 == self.__L:
                t = np.exp(z)
                suma_t = np.sum(t, axis=0, keepdims=True)
                self.__cache['A' + str(i+1)] = t / suma_t
            else:
                if self.__activation == 'sig':
                    self.__cache['A' + str(i+1)] = 1/(1 + np.exp(-z))
                else:
                    self.__cache['A' + str(i+1)] = np.tanh(z)

        return self.__cache['A' + str(i+1)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        L = -(1 / m) * np.sum(Y * np.log(A))

        return L

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.where(A < 0.5, 0, 1)

        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        limit = self.__L
        w_aux = []

        while limit > 0:
            A = cache['A' + str(limit)]
            if limit == self.__L:
                dz = A - Y
            else:
                if self.__activation == 'sig':
                    dz = np.matmul(w_aux.T, dz) * A * (1 - A)
                else:
                    dz = np.matmul(w_aux.T, dz) * (1 - A**2)
            dw = (1 / m) * np.matmul(dz, cache['A' + str(limit - 1)].T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            w_aux = self.__weights["W" + str(limit)].copy()
            self.__weights["W" + str(limit)] -= alpha * dw
            self.__weights["b" + str(limit)] -= alpha * db
            limit -= 1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
            A, cache = self.forward_prop(X)

            cost = self.cost(Y, A)
            if (verbose or graph) and i % step == 0:
                if verbose:
                    print('Cost after ' + str(i) + ' iterations: ' + str(cost))
                if graph:
                    y[i // step] = cost

            self.gradient_descent(Y, cache, alpha)

        if verbose:
            i = iterations
            print('Cost after ' + str(i) + ' iterations: ' + str(cost))
        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.suptitle("Training Cost")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.savefig("28-cost-train1.png")

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"

        dbfile = open(filename, 'ab')
        pickle.dump(self, dbfile)

    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            dbfile = open(filename, 'rb')
        except FileNotFoundError:
            return None

        if not dbfile:
            return None

        db = pickle.load(dbfile)
        dbfile.close()

        return db
