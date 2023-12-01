import os
from copy import copy

from pathlib import Path

from eventwarping.formats import setlistfile2setlistsfile
from eventwarping.eventseries import EventSeries
from eventwarping.constraints import MaxMergeSymbolConstraint, \
    NoMergeTooDistantSymbolSetConstraint, NoXorMergeSymbolSetConstraint
import matplotlib.pyplot as plt
import numpy as np

def dtw_distance(s, t, nb_events):
    d = np.ones([len(s), 3]) * np.inf
    d[0, 0] = 0
    for i in range(1, len(s)):
        for k, j in enumerate(range(i-1, i+2)):
            if i >= len(d):
                continue
            if i == j or ((4 < i%nb_events < 33) and (i%nb_events-4)//6 == (j%nb_events-4)//6):
                cost = np.abs(s[i] - t[j]) + 0.5 * np.abs(i-j)
            else:
                cost = np.inf
            d[i, k] = min(d[i - 1, k - 1] + cost,
                          d[i, k - 1] + cost,
                          d[i - 1, k] + cost)
    return d[-1, 1]


def cross_product_distance(s, t, nb_events):
    return - np.sum(s * t) - 5 * np.sum(s[:5] * t[:5])   #################


def l1_dist(s, t, nb_events):
    return - np.sum(s == t) - 5 * np.sum(s[:5] == t[:5])   #################


def distance_apnea_data(a, b):
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
    return dist


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

    setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=3)


constraints = [
    NoMergeTooDistantSymbolSetConstraint(distance_apnea_data, 5),
    MaxMergeSymbolConstraint(3),
    NoXorMergeSymbolSetConstraint([1, 2, 3, 4])
]

n = len(EventSeries().from_setlistfiles(fn_tos, window=1, intonly=True).series)
select_outer = np.array([True] * n)

for i in range(3): # number of patterns
    # do alignment
    es = EventSeries().from_setlistfiles(fn_tos, window=3, intonly=True,
                                         selected=select_outer,
                                         constraints=constraints,
                                         max_series_length=20)
    for symbol in [1, 2, 3, 4]:
        es.rescale_weights[symbol] = 5

    # remove other apnea/hypopnea such that only 1 remains per series
    es.series[:, :, 2][np.cumsum(es.series[:, :, 1], 1) == 0] = 0
    es.series[:, :, 4][np.cumsum(es.series[:, :, 3], 1) == 0] = 0
    es.series[:, ::-1, 1][np.cumsum(es.series[:, ::-1, 2], 1) == 0] = 0
    es.series[:, ::-1, 3][np.cumsum(es.series[:, ::-1, 4], 1) == 0] = 0

    n = len(es.series)
    print(n)
    for i, ws in enumerate(es.warp_yield(iterations=5, restart=True)):  ######################
        print(f"=== {i + 1:>2} ===")
        # es.plot_directions(5)

    # calculate likelihood of elements per series
    l_matrix = np.mean(es.warped_series.astype(bool), 0)
    ll = np.einsum("ijk,kj", es.warped_series.astype(bool), es.windowed_counts)   # divide by item count?
    # ll = np.einsum("ijk,kj", es.warped_series.astype(bool), l_matrix.T) + \
    #      np.einsum("ijk,kj", (1 - es.warped_series.astype(bool)), (1 - l_matrix).T)

    ll[np.sum(es.series, (1,2)) == 0] = -np.inf  ### ignore empty series

    # feature selection based on most likely (not pure clusters???)
    sel = ll > np.max([np.quantile(ll, 0.9), np.sort(ll)[-150]])
    einsum_sel = np.mean(es.warped_series[sel].astype(bool), 0)
    template = einsum_sel > 0.1   #####np.sort(einsum_sel.flatten())[-25]   #### 15 items
    max_row = np.max(einsum_sel, 0)
    template[:, :5] = True ####(np.mean(es.warped_series[sel].astype(bool), 0) > 0.8 * max_row)[:, :5]

    # calculate full dist matrix on selected and take "middle" as prototype
    m = sum(sel)
    dists = np.ones((m, m)) * np.inf
    for j in range(m):
        for k in range(m):
            series_j = copy(es.warped_series[sel][j].T.astype(bool))
            series_j[~template.T] = 0
            series_k = copy(es.warped_series[sel][k].T.astype(bool))
            series_k[~template.T] = 0
            dists[j, k] = l1_dist(series_j, series_k, es.nb_events)

    # pattern prototype based on highest ll
    id_max = np.argmax(ll)
    id_max = np.argmin(np.sort(dists, 1)[:, 20])  # prototype index
    sel2 = np.argsort(np.argsort(dists[id_max])) < 20  # closest to prototype among selected
    einsum_sel = np.mean(es.warped_series[sel][sel2].astype(bool), 0)
    template = einsum_sel > 0.1   ####np.sort(einsum_sel.flatten())[-25]   #### 15 items
    prototype = copy(es.warped_series[id_max].T.astype(bool))
    prototype[~template.T] = 0

    plt.figure()
    plt.pcolormesh(prototype)

    # distances to prototype
    dists = np.ones(n) * np.inf
    for j in range(n):
        series_j = copy(es.warped_series[j].T.astype(bool))
        series_j[~template.T] = 0
        dists[j] = cross_product_distance(series_j, prototype, es.nb_events)

    # cluster of prototype
    plt.figure()
    sel = dists < np.quantile(dists, 0.10)
    plt.pcolormesh(np.sum(es.series[sel], 0).T)
    plt.figure()
    plt.pcolormesh(np.sum(es.warped_series[sel], 0).T)

    # end loop
    select_outer[select_outer] = ~sel

es.plot_symbols()
plt.show()

print(1)
