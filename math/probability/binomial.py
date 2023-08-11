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

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """"Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        div = (2 * pi * self.stddev**2)**(1 / 2)

        return (e**((-(x - self.mean)**2)/(2 * self.stddev**2))) / div

    def erf(self, x):
        """error function"""
        pi = 3.1415926536

        return (x - x**3/3 + x**5/10 - x**7/42 + x**9/216) * 2 / pi**(1/2)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        return (1 + self.erf((x - self.mean) / (self.stddev * 2**(1/2)))) / 2
