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
            self.lambtha = sum(data) / len(data)

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