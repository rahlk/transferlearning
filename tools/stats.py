from __future__ import division
import numpy as np
from pdb import set_trace

class Counter:
    def __init__(self, before, after, indx):
        self.indx = indx
        self.actual = before
        self.predicted = after
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        for a, b in zip(self.actual, self.predicted):
            if a == indx and b == indx:
                self.TP += 1
            elif a == b and a != indx:
                self.TN += 1
            elif a != indx and b == indx:
                self.FP += 1
            elif a == indx and b != indx:
                self.FN += 1
            elif a != indx and b != indx:
                pass

    def stats(self):
        try:
            Sen = self.TP / (self.TP + self.FN)
            Spec = self.TN / (self.TN + self.FP)
            Prec = self.TP / (self.TP + self.FP)
            Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
            F = 2 * (Prec * Sen) / (Prec + Sen)
            G = 2 * Sen * Spec / (Sen + Spec)
            # F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
            # G1 = Sen * Spec / (Sen + Spec)
            ED = np.sqrt(0.6*(1-Sen)**2+0.3*(1-Spec)**2)
            # set_trace()
            return Sen, 1-Spec, Prec, Acc, F, G, ED
        except ZeroDivisionError:
            return 0, 0, 0, 0, 0, 0


class ABCD:
    """ Statistics Stuff, confusion matrix, all that jazz...
    """

    def __init__(self, before, after):
        self.actual = before
        self.predicted = after

    def __call__(self):
        uniques = set(self.actual)
        for u in list(uniques):
            yield Counter(self.actual, self.predicted, indx=u)
