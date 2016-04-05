from __future__ import print_function, division

from pdb import set_trace

from Oracle.methods1 import explore
from Planners.xtree import xtree
from tools.sk import rdivDemo

__author__ = 'rkrsn'

# from Oracle.Prediction import rforest


def improve():
    e = []
    train, test = explore(dir='Data/Jureczko/')
    for a, b in zip(train, test):
        new = xtree(train=a, test=b)
        new.to_csv(index=False)
        set_trace()
        rdivDemo(e, isLatex=False, globalMinMax=True, high=100, low=0)


if __name__ == '__main__':
    improve()
