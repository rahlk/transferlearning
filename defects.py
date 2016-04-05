from __future__ import print_function, division

__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.tune.dEvol import tuner
from tools.oracle import *
# Timing
from time import time
import numpy as np
# Parallelism
from multiprocessing import Pool


class temporal:
    def __init__(self):
        pass

    def deltas(self):
        from collections import Counter
        counts = {}

        def save2plot(header, counts, labels, N):
            for h in header: say(h + ' ')
            print('')
            for l in labels:
                say(l[1:] + ' ')
                for k in counts.keys():
                    say("%0.2f," % (counts[k][l] * 100 / N))
                print('')

        for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
            print("##", name)
            e = []
            train, test = explore(dir='Data/Jureczko/', name=name)
            for planners in [xtree, method1]:  # , method2, method3]:
                # aft = [planners.__doc__]
                for _ in xrange(10):
                    keys = []
                    everything, changes = planners(train, test, justDeltas=True)
                    for ch in changes: keys.extend(ch.keys())
                    counts.update({planners.__doc__: Counter(keys)})
            header = ['Features'] + counts.keys()
            save2plot(header, counts, everything, N=len(changes))

    def deltas0(self):
        from collections import Counter
        counts = {}

        def save2plot(header, counts, labels, numCh, N):
            #   for h in header: say(h+' ')
            print('')
            for id, l in enumerate(labels):
                # say(l[1:]+' ')
                #   for k in counts.keys():
                # print(counts[k])
                try:
                    if numCh[l] / N > 0:
                        # set_trace()
                        say(l[1:] + " %d %0.2f %0.2f %0.2f" % (
                        id + 1, np.median(counts[l]), np.percentile(counts[l], 25), np.percentile(counts[l], 75)))
                    else:
                        say(l[1:] + " %d 0 0 0" % (id + 1))

                except:
                    # set_trace()
                    say(l[1:] + " %d 0 0 0" % (id + 1))
                # set_trace()
                print('')
                # set_trace()

            print("")
            for id, l in enumerate(labels):
                say(l[1:] + " %d %0.2f " % (id + 1, len(counts[l]) * 100 / N))
                print('')

        for name in ['jedit']:
            print("##", name)
            e = []
            train, test = explore(dir='Data/Jureczko/', name=name)
            for planners in [xtree]:
                # keys=[]
                everything, changes = planners(train, test, justDeltas=True)
                keys = everything.values.tolist()
                # set_trace()
                for k in keys: counts.update({k: []})
                f_chCount = []
                for _ in xrange(10):
                    everything, changes = planners(train, test, justDeltas=True)
                    for c in changes:
                        for key in c.keys():
                            f_chCount.append(key)
                            counts[key].append(c[key])

            # set_trace()
            header = ['Features'] + counts.keys()
            save2plot(header, counts, everything, numCh=Counter(f_chCount), N=10 * len(changes))

    def improve(self):
        for name in ['ant', 'lucene', 'poi', 'ivy', 'jedit']:
            print("##", name)
            e = []
            train, test = explore(dir='Data/Jureczko/', name=name)
            for planners in [xtree]:  # , method1, method2, method3]:
                aft = [planners.__doc__]
                for _ in xrange(10):
                    aft.append(planners(train, test, justDeltas=False))
                    # set_trace()
                e.append(aft)
            try:
                rdivDemo(e, isLatex=False, globalMinMax=True, high=100, low=0)
            except:
                set_trace()


def parCross(indx):
    """
    Learn from all other projects. Compare the results.
    :return:
    """
    # names=['ant', 'ivy', 'jedit', 'lucene', 'poi']
    train, test = explore(dir='Data/Jureczko/')
    one = test[indx]
    e = []
    for two in train:
        # say(two[0].split('/')[-2]+" - "+one[0].split('/')[-2])
        # print("##", "Train: ", two[0].split('/')[-2],"Test: ", one[0].split('/')[-2])
        aft = [two[0].split('/')[-2] + " - " + one[0].split('/')[-2]]
        rfTrain = train[indx]
        # params = None
        params = tuner(rfTrain)
        # print("Tuning time: %0.2f"%(time()-t))
        t = time()
        for _ in xrange(10):
            _, new = xtree(two, one, rftrain=rfTrain
                           , tunings=params, justDeltas=False)
            aft.append(new)
        # print("Average Planning time: %0.2f"%((time()-t)/1))
        e.append(aft)
    rdivDemo(e)


class cross:
    def __init__(self):
        pass

    def improve1(self, indx):
        """
        Learn from all other projects. Compare the results.
        :return:
        """
        # names=['ant', 'ivy', 'jedit', 'lucene', 'poi']
        train, test = explore(dir='Data/Jureczko/')
        one = test[indx]
        e = []
        for two in train:
            print("##", "Train: ", two[0].split('/')[-2], "Test: ", one[0].split('/')[-2])
            aft = [two[0].split('/')[-2]]
            rfTrain = train[indx]
            set_trace()
            t = time()
            # params = None
            params = tuner(rfTrain)
            # print("Tuning time: %0.2f"%(time()-t))
            t = time()
            for _ in xrange(1):
                _, new = planners(two, one, rftrain=rfTrain
                                  , tunings=params, justDeltas=False)
                aft.append(new)
            # print("Average Planning time: %0.2f"%((time()-t)/1))
            e.append(aft)
        rdivDemo(e)

    def deltas(self):
        from collections import Counter

        def save2plot(header, counts, labels, N):
            for h in header: say(h + ' ')
            print('')
            for l in labels:
                say(l[1:] + ' ')
                for k in counts.keys():
                    say("%0.2f " % (counts[k][l] * 100 / N))
                print('')

        names = ['ant', 'ivy', 'jedit', 'lucene', 'poi']
        for planners in [xtree]:  # , method1, method2, method3]:
            counts = {}
            for one in names:
                e = []
                for two in names:
                    # print("##", two, one)
                    aft = [two]
                    train, _ = explore(dir='Data/Jureczko/', name=two)
                    test, _ = explore(dir='Data/Jureczko/', name=one)
                    # for _ in xrange(1):
                    keys = []
                    everything, changes = planners(train, test, justDeltas=True)
                    attr = list(set([c.keys() for c in changes]))
                    for ch in changes: counts.update
                    for ch in changes: keys.extend(ch.keys())
                    counts.update({two: Counter(keys)})
                    # set_trace()
                header = ['Features'] + counts.keys()
                save2plot(header, counts, everything, N=len(changes))


class accuracy:
    """
    Test prediction accuracy with RF
    """

    def __init__(i):
        pass

    def RF(i):
        train, test = explore(dir='Data/Jureczko/')
        # set_trace()
        print("Data,\t Pd,\t Pf,\t Pd,\t Pf")
        print("\t,Naive,\t\t,Tuned,\t,\t")
        # print("# %s"%(te[0].split('/')[-2]))
        for tr, te in zip(train, test):
            E0, E1 = [], []
            # for tr in train:
            Pd = [tr[0].split('/')[-2]]
            Pf = [tr[0].split('/')[-2]]
            G = [tr[0].split('/')[-2]]
            tunings = tuner(tr, smoteTune=True)
            for _ in xrange(1):
                # TUNE RF + Naive SMOTE

                actual, preds = rforest(tr, te, tunings=tunings, smoteit=True, smoteTune=False)
                abcd = ABCD(before=actual, after=preds)
                F = np.array([k.stats()[-2] for k in abcd()])
                Pd0 = np.array([k.stats()[0] for k in abcd()])
                Pf0 = np.array([k.stats()[1] for k in abcd()])
                G.append(F[0])
                say(tr[0].split('/')[-2] + '\t, %0.2f,\t %0.2f,\t' % (Pd0[1], 1 - Pf0[1]))

                # Tune+SMOTE
                smote = True
                actual, preds = rforest(tr, te, tunings=tunings, smoteit=True)
                abcd = ABCD(before=actual, after=preds)
                F = np.array([k.stats()[-2] for k in abcd()])
                Pd0 = np.array([k.stats()[0] for k in abcd()])
                Pf0 = np.array([k.stats()[1] for k in abcd()])
                # set_trace()
                G.append(F[0])
                say('%0.2f,\t %0.2f\n' % (Pd0[1], 1 - Pf0[1]))

                # Pd.append(Pd0)
                # Pf.append(Pf0)
                # E0.append(G)
                # E1.append(Pf)
                # rdivDemo(E0)
                # rdivDemo(E1)
        set_trace()

    def SVM(i):
        train, test = explore(dir='Data/Jureczko/')
        # set_trace()
        print("Data, Pd, Pf")
        # print("# %s"%(te[0].split('/')[-2]))
        for tr, te in zip(train, test):
            E0, E1 = [], []
            # for tr in train:
            Pd = [tr[0].split('/')[-2]]
            Pf = [tr[0].split('/')[-2]]
            G = [tr[0].split('/')[-2]]
            # for _ in xrange(1):
            tunings = None  # tuner(tr)
            smote = True
            # tunings = tuner(tr)
            # set_trace()
            actual, preds = rforest(tr, te, tunings=tunings, smoteit=smote)
            abcd = ABCD(before=actual, after=preds)
            F = np.array([k.stats()[-2] for k in abcd()])
            Pd0 = np.array([k.stats()[0] for k in abcd()])
            Pf0 = np.array([k.stats()[1] for k in abcd()])
            # set_trace()
            G.append(F[0])
            say(tr[0].split('/')[-2] + ', %0.2f, %0.2f\n' % (Pd0[1], 1 - Pf0[1]))
            #
            # # Tune+SMOTE
            # tunings = None#tuner(tr)
            # smote= True
            # actual, preds = rforest(tr, te, tunings=tunings, smoteit=smote)
            # abcd = ABCD(before=actual, after=preds)
            # F = np.array([k.stats()[-2] for k in abcd()])
            # Pd0 = np.array([k.stats()[0] for k in abcd()])
            # Pf0 = np.array([k.stats()[1] for k in abcd()])
            # # set_trace()
            # G.append(F[0])
            # say(', %0.2f, %0.2f\n'%(Pd0[1], 1-Pf0[1]))

            # Pd.append(Pd0)
            # Pf.append(Pf0)
            # E0.append(G)
            # E1.append(Pf)
            # rdivDemo(E0)
            # rdivDemo(E1)
        set_trace()


class mccabe:
    """
    MacCabe Halsted dataset
    """

    def __init__(self):
        pass

    def deltas(self):
        from collections import Counter
        counts = {}

        def save2plot(header, counts, labels, N):
            for h in header: say(h + ' ')
            print('')
            for l in labels:
                say(l[1:] + ' ')
                for k in counts.keys():
                    say("%0.2f " % (counts[k][l] * 100 / N))
                print('')

        for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
            print("##", name)
            e = []
            train, test = explore(dir='Data/mccabe/', name=name)
            for planners in [xtree, method1, method2, method3]:
                # aft = [planners.__doc__]
                for _ in xrange(1):
                    keys = []
                    everything, changes = planners(train, test, justDeltas=True)
                    for ch in changes: keys.extend(ch.keys())
                    counts.update({planners.__doc__: Counter(keys)})
            header = ['Features'] + counts.keys()
            save2plot(header, counts, everything, N=len(changes))

    def improve(self):
        for name in ['jm', 'cm', 'ar', 'kc', 'jm', 'mw']:
            print("##", name)
            train, test = explore(dir='Data/mccabe/', name=name)
            all = train + test
            for test in all:
                if not test == train:
                    e = []
                    for planners in [xtree, method1, method2, method3]:
                        aft = [planners.__doc__]
                        for _ in xrange(1):
                            aft.append(
                                planners(train=[aa for aa in all if not aa == test], test=[test], justDeltas=False))
                            # set_trace()
                        e.append(aft)
                    rdivDemo(e)

    def acc(self):
        for name in ['MC']:  # , 'kc', 'cm', 'ar', 'jm', 'mw']:
            print("##", name)
            train, test = explore(dir='Data/mccabe/', name=name)
            # set_trace()
            all = train + test
            E0 = []
            for test in all:
                if not test == train:
                    # set_trace()
                    G = [test.split('/')[-1][:-4]]
                    actual, preds = rforest(train=[aa for aa in all if not aa == test], test=[test])
                    abcd = ABCD(before=actual, after=preds)
                    F = np.array([k.stats()[-1] for k in abcd()])
                    # set_trace()
                    G.append(F[0])
                E0.append(G)
            rdivDemo(E0)


def parallel():
    collect = []
    train, test = explore(dir='/share/rkrish11/Datasets/Jureczko/')
    n_proc = len(train)
    p = Pool(processes=n_proc)
    # planner = cross().improve1
    # set_trace()
    collect.append(p.map(parCross, range(n_proc)))
    for cc in collect: print(cc)
    set_trace()


if __name__ == '__main__':
    # accuracy().SVM()
    # accuracy().RF()
    # parallel()
    temporal().deltas0()
    # temporal().deltas()
    # cross().improve1()
    # mccabe().improve()
    # mccabe().acc()
    # temporal().improve()
    # cross().deltas()
