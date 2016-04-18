from __future__ import print_function, division

from os import walk
from pdb import set_trace
from scipy.stats import kruskal as HTest
from pandas import *

__author__ = 'rkrsn'


def explore(dir):
    datasets = []
    for (dirpath, dirnames, filenames) in walk(dir):
        datasets.append(dirpath)

    training = []
    testing = []
    for k in datasets[1:]:
        train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
        test = [train[0][0] + '/' + train[0][1].pop(-1)]
        training.append(
            [train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store'])
        testing.append(test)
    return training, testing


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
    bellwether = train.pop(-5) + test.pop(-5)
    all = [t + tt for t, tt in zip(train, test)]
    bw = csv2DF(bellwether)
    col = bw.columns
    # print(' ,' + ','.join(col.values.tolist()[:-1]))
    for dat in all:
        me = csv2DF(dat)
        print(dat[0].split('/')[-2],end="")
        val =[]
        for k in col[:-1]:
            h,p  = HTest(me[k].values, bw[k].values)
            val.append(int(p>=0.05))
        print(',%d/%d'%(sum(val), (len(col)-1)))
    print("")


if __name__=="__main__":
    kruskalWallis(proj='jur')
    # kruskalWallis(proj='mccabe')
    # kruskalWallis(proj='aeeem')
    # kruskalWallis(proj='Relink')