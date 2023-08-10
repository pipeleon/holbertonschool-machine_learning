#!/usr/bin/env python3
"""Task 3 Project Probability"""


class Exponential():
    """Class that represents a Exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Init method"""
        if data is None:
            self.lambtha = lambtha
        else:
            self.data = data
            self.lambtha = len(data) / sum(data)

    @property
    def lambtha(self):
        """lambtha property"""
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, lambtha):
        """Setter lambtha"""
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = lambtha

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

    def pdf(self, x):
        """Calculates the value of the PDF  for a given time period"""
        if x < 0:
            return 0

        e = 2.7182818285
        k = int(x)

        return e**(-self.lambtha * x) * self.lambtha

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        if x < 0:
            return 0

        e = 2.7182818285
        k = int(x)

        return 1 - e**(-self.lambtha * x)
