#!/usr/bin/env python3
"""Task 6 Project Probability"""


class Normal():
    """Class that represents a Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Init method"""
        if data is None:
            self.mean = mean
            self.stddev = stddev
        else:
            self.data = data
            self.mean = sum(data) / len(data)
            sqr_sum = 0
            for i in data:
                sqr_sum += (i - self.mean)**2
            self.stddev = (sqr_sum / len(data))**(1/2)

    @property
    def mean(self):
        """mean property"""
        return self.__mean

    @mean.setter
    def mean(self, mean):
        """Setter mean"""
        self.__mean = mean

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
    def stddev(self):
        """stddev property"""
        return self.__stddev

    @stddev.setter
    def stddev(self, stddev):
        """Setter stddev"""
        if stddev <= 0:
            raise ValueError("stddev must be a positive value")
        self.__stddev = stddev

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
