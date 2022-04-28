import numpy as np


def tanh():
    return Activation(np.tanh, lambda x: 1.0 - x ** 2)


def sigmoid():
    return Activation(lambda x: 1.0 / (1.0 + np.exp(-x)), lambda x: x * (1.0 - x))


class Activation(object):
    def __init__(self, acti, dacti):
        self.__acti = acti
        self.__dacti = dacti

    def fn(self, x):
        return self.__acti(x)

    def dfn(self, x):
        return self.__dacti(x)
