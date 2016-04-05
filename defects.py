from __future__ import print_function, division

from pdb import set_trace
from os import remove
from Oracle.Prediction import Bugs, rforest
from Oracle.methods1 import explore, createTbl
from Planners.xtree import xtree
from tools.sk import rdivDemo

__author__ = 'rkrsn'


def improve():
    all = []
    train, test = explore(dir='Data/Jureczko/')
    for a, b in zip(train, test):
        me = [test[0][0].split('/')[-2]]
        for _ in xrange(1):
            new = xtree(train=a, test=b)
            new.to_csv('new.csv', index=False)
            rfTrain = createTbl(a)
            rfTest = createTbl(['new.csv'])
            actual = Bugs(createTbl(b))
            imporved = rforest(train=rfTrain, test=rfTest)
            gain = float("%0.2f"%((1 - sum(imporved) / sum(actual)) * 100))
            me.append(gain)
            remove('new.csv')
        all.append(me)
    rdivDemo(all, isLatex=False, globalMinMax=True, high=100, low=0)


if __name__ == '__main__':
    improve()
