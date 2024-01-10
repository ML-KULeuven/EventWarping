import os
from copy import copy

from pathlib import Path

from eventwarping.formats import setlistfile2setlistsfile
from eventwarping.eventseries import EventSeries
from eventwarping.constraints import MaxMergeSymbolConstraint, \
    NoMergeTooDistantSymbolSetConstraint, NoXorMergeSymbolSetConstraint
import matplotlib.pyplot as plt
import numpy as np

def cross_product_distance(s, t):
    return - np.sum(s * t) - 5 * np.sum(s[:5] * t[:5])   #################


def l1_dist(s, t):
    return - np.sum(s == t) - 5 * np.sum(s[:5] == t[:5])   #################


def dist_respiratory_itemsets(a, b, d_empty=0):
    # sum of the maximal distance in peak heights per signal between itemsets a and b
    if a is None:
        return 0
    dist = 0
    for k in range(4):  # signals
        l = 5 + 7 * k  # first of signal
        items_in_signal = set(range(l, l + 7))
        a_signal = a.intersection(items_in_signal)
        b_signal = b.intersection(items_in_signal)
        if len(a_signal) > 0 and len(b_signal) > 0:
            dist += np.max([np.abs(m - n)
                            for m in a_signal
                            for n in b_signal], initial=0)
        elif len(a_signal) > 0 or len(b_signal) > 0:
            dist += d_empty
    return dist


def dist_respiratory_series(s, t, d_empty=0):
    d = 0
    for j in range(s.shape[1]):
        d += dist_respiratory_itemsets(set(np.argwhere(s[:, j])[:,0]), set(np.argwhere(t[:, j])[:,0]), d_empty)
    d += 5 * np.sum(s[:5] != t[:5])
    d += np.sum(s[33:] != t[33:])
    return d


# folder where apnea data is stored
folder = Path(r"C:\Users\dries\input_data_pattern_mining6")
fns = [i for i in os.listdir(folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))][::25]
fns = [folder / i for i in fns]

fn_tos = []
for fn in fns:
    print(f"Path exists = {fn.exists()}: {fn}")
    fn_to = fn.parent / (fn.stem + "_series.txt")
    print(f"Saving to {fn_to}")
    fn_tos += [fn_to]

    setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=5)

constraints = [
    NoMergeTooDistantSymbolSetConstraint(dist_respiratory_itemsets, 5),
    MaxMergeSymbolConstraint(5),
    NoXorMergeSymbolSetConstraint([1, 2, 3, 4])
]

n = len(EventSeries().from_setlistfiles(fn_tos, window=1, intonly=True).series)
select_outer = np.array([True] * n)

for i in range(3): # number of patterns
    # do alignment
    es = EventSeries().from_setlistfiles(fn_tos, window=5, intonly=True,   ####window=3
                                         selected=select_outer,
                                         constraints=constraints,
                                         max_series_length=25)
    for symbol in [1, 2, 3, 4]:
        es.rescale_weights[symbol] = 5

    es.insert_spacers(1)   ##########

    # remove other apnea/hypopnea such that only 1 remains per series
    es.series[:, :, 2][np.cumsum(es.series[:, :, 1], 1) == 0] = 0
    es.series[:, :, 4][np.cumsum(es.series[:, :, 3], 1) == 0] = 0
    es.series[:, ::-1, 1][np.cumsum(es.series[:, ::-1, 2], 1) == 0] = 0
    es.series[:, ::-1, 3][np.cumsum(es.series[:, ::-1, 4], 1) == 0] = 0

    # no difference between hypopnea and apnea
    es.series[:, :, 1] += es.series[:, :, 3]
    es.series[:, :, 2] += es.series[:, :, 4]
    es.series[:, :, 3] = 0
    es.series[:, :, 4] = 0

    n = len(es.series)
    print(n)
    for i, ws in enumerate(es.warp_yield(iterations=10, restart=True)):  ######################
        print(f"=== {i + 1:>2} ===")
        # es.plot_directions(5)

    # fraction occurrences of items per timestamp
    prob_items_per_set = es.get_counts(ignore_merged=True) / es.nb_series

    # mask indicating all infrequent items
    mask = prob_items_per_set.T > 0.01

    # calculate likelihood of elements per series in the mask
    ll = np.einsum("ijk,kj", es.warped_series.astype(bool) * mask, prob_items_per_set)   # divide by item count?
    # ll = np.einsum("ijk,kj", es.warped_series.astype(bool), l_matrix.T) + \
    #      np.einsum("ijk,kj", (1 - es.warped_series.astype(bool)), (1 - l_matrix).T)

    # calculate the likelihood of the 20 most likely items in the mask
    ll = np.sum(np.log(np.sort((es.warped_series.astype(bool) * mask * prob_items_per_set.T).reshape((-1, 49 * 39)), axis=1)[:, -20:]), axis=1)   ### count > 1??? If < 20 nonzero???

    ll[np.sum(es.series, (1,2)) == 0] = -np.inf  ### ignore empty series

    # feature selection based on most likely (not pure clusters???)
    sel = ll > np.max([np.quantile(ll, 0.9), np.sort(ll)[-150]])
    einsum_sel = np.mean(es.warped_series[sel].astype(bool), 0)
    template = einsum_sel > 0.25
    max_row = np.max(einsum_sel, 0)
    template[:, :5] = True

    # calculate full dist matrix on selected and take "middle" as prototype
    m = sum(sel)
    dists1 = np.ones((m, m)) * np.inf
    for j in range(m):
        for k in range(m):
            series_j = copy(es.warped_series[sel][j].T.astype(bool))
            series_j[~template.T] = 0
            series_k = copy(es.warped_series[sel][k].T.astype(bool))
            series_k[~template.T] = 0
            dists1[j, k] = dist_respiratory_series(series_j, series_k, 1)  ####### default if one out of template

    # pattern prototype based on highest ll
    id_max = np.argmax(ll)
    id_max = np.argmin(np.sort(dists1, 1)[:, 20])  # prototype index
    sel2 = np.argsort(np.argsort(dists1[id_max])) < 20  # closest to prototype among selected
    einsum_sel = np.mean(es.warped_series[sel][sel2].astype(bool), 0)
    template = einsum_sel > 0.25
    template[:, :5] = True
    prototype = copy(es.warped_series[id_max].T.astype(bool))
    prototype[~template.T] = 0

    plt.figure()
    plt.pcolormesh(prototype)

    # distances to prototype
    dists = np.ones(n) * np.inf
    for j in range(n):
        series_j = copy(es.warped_series[j].T.astype(bool))
        series_j[~template.T] = 0
        dists[j] = dist_respiratory_series(series_j, prototype)

    # cluster of prototype
    plt.figure()
    sel = dists < np.quantile(dists, 0.05)
    plt.pcolormesh(np.sum(es.series[sel], 0).T)
    plt.figure()
    plt.pcolormesh(np.sum(es.warped_series[sel], 0).T)

    # end loop
    select_outer[select_outer] = ~sel

es.plot_symbols()
plt.show()

print(1)
