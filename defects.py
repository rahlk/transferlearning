from __future__ import print_function, division

from os import remove
from pdb import set_trace
from pandas import *

from Oracle.Prediction import Bugs, rforest
from Oracle.methods1 import explore, createTbl
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.stats import ABCD

__author__ = 'rkrsn'


def csv2DF(dir, as_mtx=False, toBin=False):
    files = []
    for f in dir:
        df = read_csv(f)
        headers = [h for h in df.columns if '?' not in h]
        if isinstance(df[df.columns[-1]][0], str):
            df[df.columns[-1]] = DataFrame([0 if 'N' in d or 'n' in d else 1 for d in df[df.columns[-1]]])
        if toBin:
            df[df.columns[-1]] = DataFrame([1 if d > 0 else 0 for d in df[df.columns[-1]]])
        files.append(df[headers])
    "For N files in a project, use 1 to N-1 as train."
    data_DF = concat(files)
    if as_mtx:
        return data_DF.as_matrix()
    else:
        return data_DF


def kruskalWallis(proj='jur'):
    if proj == 'jur':
        dir = 'Data/Jureczko/'
    elif proj == 'mccabe':
        dir = 'Data/mccabe/'
    elif proj == 'aeeem':
        dir = 'Data/AEEEM/'
    elif proj == 'Relink':
        dir = 'Data/relink/'

    train, test = explore(dir)
    bellwether = train.pop(-2) + test.pop(-2)
    all = [t + tt for t, tt in zip(train, test)]
    for dat in all:
        me = csv2DF(dat)
        bw = csv2DF(bellwether)
        set_trace()

def improve(proj='jur'):
    if proj == 'jur':
        dir = 'Data/Jureczko/'
    elif proj == 'mccabe':
        dir = 'Data/mccabe/'
    elif proj == 'aeeem':
        dir = 'Data/AEEEM/'
    elif proj == 'Relink':
        dir = 'Data/relink/'

    train, test = explore(dir)
    bellwether = train.pop(-2) + test.pop(-2)

    for a, b in zip(train, test):
        all = []
        me = [b[0].split('/')[-2]]
        luc = [bellwether[0].split('/')[-2]]
        for _ in xrange(4):
            # Train on historical data
            new = xtree(train=a, test=b)
            new.to_csv('me.csv', index=False)
            rfTrain = createTbl(a, isBin=True)
            rfTest = createTbl(['me.csv'])
            actual = Bugs(createTbl(b, isBin=True))
            imporved = rforest(train=rfTrain, test=rfTest)
            gain = float("%0.2f" % ((1 - sum(imporved) / sum(actual)) * 100))
            me.append(gain)
            remove('me.csv')

            # Train on lucene
            new = xtree(train=bellwether, test=b)
            new.to_csv('bellwether.csv', index=False)
            rfTrain = createTbl(a, isBin=True)
            rfTest = createTbl(['bellwether.csv'])
            actual = Bugs(createTbl(b, isBin=True))
            imporved = rforest(train=rfTrain, test=rfTest)
            gain = float("%0.2f" % ((1 - sum(imporved) / sum(actual)) * 100))
            luc.append(gain)
            remove('bellwether.csv')

        all.extend([me, luc])
        rdivDemo(all, isLatex=False, globalMinMax=True, high=100, low=0)


def test_oracle(proj="jur"):
    if proj == 'jur':
        dir = 'Data/Jureczko/'
    elif proj == 'mccabe':
        dir = 'Data/mccabe/'
    elif proj == 'aeeem':
        dir = 'Data/AEEEM/'
    elif proj == 'Relink':
        dir = 'Data/relink/'

    train, test = explore(dir)
    bellwether = train.pop(5) + test.pop(5)
    for a, b in zip(train, test):
        me = [b[0].split('/')[-2]]
        oth = [bellwether[0].split('/')[-2]]

        for _ in xrange(10):
            # Train on historical data
            rfTrain = createTbl(a, isBin=True)
            rfTest = createTbl(b, isBin=True)
            actual = Bugs(rfTest)
            prdctd = rforest(train=rfTrain, test=rfTest)
            abcd = ABCD(before=actual, after=prdctd)
            ED = np.array([k.stats()[-1] for k in abcd()])
            me.append(ED[1])
            # print(b[0].split('/')[-2], "\nPd = %0.2f , Pf = %0.2f " % (Pd[1], Pf[1]))

            # Train on bellwether
            rfTrain = createTbl(bellwether, isBin=True)
            rfTest = createTbl(b, isBin=True)
            actual = Bugs(rfTest)
            prdctd = rforest(train=rfTrain, test=rfTest)
            abcd = ABCD(before=actual, after=prdctd)
            ED = np.array([k.stats()[-1] for k in abcd()])
            oth.append(ED[1])
            # print("Bellwether", "\nPd = %0.2f , Pf = %0.2f\n" % (Pd[1], Pf[1]))

        E = [me, oth]
        try:
            rdivDemo(E, isLatex=True, globalMinMax=True, high=1, low=0)
        except:
            set_trace()


def _test_improve():
    # print('#### Jureczko\n```')
    # improve(proj='jur')
    # print('```')
    #
    print('#### McCabes\n```')
    improve(proj='mccabe')
    print('```')

    # print('#### AEEEM\n```')
    # improve(proj='aeeem')
    # print('```')
    #
    # print('#### ReLink\n```')
    # improve(proj='relink')
    # print('```')


def _test_oracle():
    print('#### Jureczko\n```')
    test_oracle(proj='jur')
    print('```')
    #
    # print('#### McCabes\n```')
    # test_oracle(proj='mccabe')
    # print('```')
    #
    # print('#### AEEEM\n```')
    # test_oracle(proj='aeeem')
    # print('```')
    #
    # print('#### ReLink\n```')
    # test_oracle(proj='relink')
    # print('```')


if __name__ == '__main__':
    _test_oracle()

    # _test_improve()
