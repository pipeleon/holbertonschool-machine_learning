#!/usr/bin/env python3
"""Task 10 Project Probability"""


class Binomial():
    """Class that represents a Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Init method"""
        if data is None:
            self.n = n
            self.p = p
        else:
            self.data = data
            mean = sum(data) / len(data)
            sqr_sum = 0
            for i in data:
                sqr_sum += (i - mean)**2
            variance = (sqr_sum / len(data))
            p_temp = 1 - variance / mean
            self.n = round(sum(map(lambda x: x / p_temp, data)) / len(data))
            self.p = sum(map(lambda x: x / self.n, data)) / len(data)

    @property
    def n(self):
        """n property"""
        return self.__n

    @n.setter
    def n(self, n):
        """Setter n"""
        if n <= 0:
            raise ValueError("n must be a positive value")
        self.__n = n

    @property
    def data(self):
        """data property"""
        return self.__data

    @data.setter
    def data(self, data):
        """Setter data"""
        if type(data) is not list:
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        self.__data = data

    @property
    def p(self):
        """p property"""
        return self.__p

    @p.setter
    def p(self, p):
        """Setter p"""
        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
        self.__p = p

    def fc(self, x):
        """Funtion for fc"""
        fact = 1

        for i in range(1, x+1):
            fact = fact * i

        return fact

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if k < 0:
            return 0

        k = int(k)
        pi = 3.1415926536
        div = (self.fc(k) * self.fc(self.n - k))

        return self.fc(self.n) * self.p**k * (1 - self.p)**(self.n - k) / div

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if k < 0:
            return 0

        k = int(k)
        acumulative = 0

        for i in range(0, k+1):
            acumulative += self.pmf(i)

        return acumulative
