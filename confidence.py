from numpy import *
import scipy as sp
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * array(data)
    n = len(a)
    m, se = mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

