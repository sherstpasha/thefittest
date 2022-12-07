import numpy as np


def rank_scale_data(arr):
    arr = arr.copy()
    raw_ranks = np.zeros(shape=(arr.shape[0]))
    argsort = np.argsort(arr)

    s = (arr[:, np.newaxis] == arr).astype(int)
    raw_ranks[argsort] = np.arange(arr.shape[0]) + 1
    ranks = np.sum(raw_ranks*s, axis=1)/s.sum(axis=0)
    return ranks/ranks.sum()


def scale_data(arr):
    arr = arr.copy()
    max_ = arr.max()
    min_ = arr.min()
    if max_ == min_:
        arr_n = np.ones_like(arr)
    else:
        arr_n = (arr - min_)/(max_ - min_)
    return arr_n


class SamplingGrid:

    def __init__(self, left, right, parts):
        self.left = left
        self.right = right
        self.parts = parts
        self.h = np.abs(left - right)/(2.0**parts - 1)

    def decoder(self, population_parts, left_i, h_i):
        ipp = population_parts.astype(int)
        int_convert = np.sum(ipp*(2**np.arange(ipp.shape[1],
                                               dtype=int)),
                             axis=1)
        return left_i + h_i*int_convert

    def transform(self, population):
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)
        fpp = [self.decoder(p_parts_i, left_i, h_i)
               for p_parts_i, left_i, h_i in zip(p_parts,
                                                 self.left,
                                                 self.h)]
        return np.vstack(fpp).T
