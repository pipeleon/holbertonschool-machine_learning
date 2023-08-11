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
