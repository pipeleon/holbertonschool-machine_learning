#!/usr/bin/env python3
"""Task 0 Project Probability"""


class Poisson():
    """Class that represents a poisson distribution"""
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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if k < 0:
            return 0

        e = 2.7182818285
        k = int(k)
        fact = 1

        for i in range(1, k+1):
            fact = fact * i

        return e**(-self.lambtha) * self.lambtha**k / fact

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if k < 0:
            return 0

        k = int(k)
        acumulative = 0

        for i in range(0, k+1):
            acumulative += self.pmf(i)

        return acumulative
