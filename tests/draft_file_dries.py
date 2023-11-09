import os
from copy import copy

from pathlib import Path
from eventwarping.formats import setlistfile2setlistsfile
from eventwarping.eventseries import EventSeries
from eventwarping.constraints import ApneaConstraints
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
    return - sum(s*t)

def l1_dist(s, t, nb_events):
    return - sum(np.abs(s-t))

# folder where apnea data is stored
folder = Path(r"C:\Users\dries\input_data_pattern_mining6")
fn1 = folder / "MST_M03_0004086_itemsets.txt"  # Three occurrences
fn2 = folder / "DZ_00001_0000207#01_itemsets.txt"  # 146 occurrences
fn3 = folder / "DZ_DZ1_0000003#01_itemsets.txt"  # One occurrence
fns = [fn1, fn2, fn3]

# fns = [i for i in os.listdir(folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))][::5]
# fns = [folder / i for i in fns]

fn_tos = []
for fn in fns:
    print(f"Path exists = {fn.exists()}: {fn}")
    fn_to = fn.parent / (fn.stem + "_series.txt")
    print(f"Saving to {fn_to}")
    fn_tos += [fn_to]

    setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=3)


es = EventSeries().from_setlistfiles(fn_tos, window=3, intonly=True,
                                         constraints=ApneaConstraints, max_series_length=20)
n = len(es.series)
select_outer = np.array([True] * n)

for i in range(3): # number of patterns
    # do alignment
    es = EventSeries().from_setlistfiles(fn_tos, window=3, intonly=True,
                                         selected=select_outer,
                                         constraints=ApneaConstraints,
                                         max_series_length=20)
    n = len(es.series)
    print(n)
    for i, ws in enumerate(es.warp_yield(iterations=10, restart=True)):
        print(f"=== {i + 1:>2} ===")

    # calculate likelihood of elements per series
    ll = np.einsum("ijk,kj", es.series, es.windowed_counts)

    # feature selection based on most likely (not pure clusters???)
    sel = ll > np.quantile(ll, 0.9)
    einsum_sel = np.mean(es.series[sel], 0)
    template = einsum_sel > 0.15
    max_row = np.max(einsum_sel, 0)
    template[:, :5] = (np.mean(es.series[sel], 0) > 0.8 * max_row)[:, :5]

    # # calculate full dist matrix on selected and take "middle" as prototype
    # m = sum(sel)
    # dists = np.ones((m, m)) * np.inf
    # for j in range(m):
    #     for k in range(m):
    #         series_j = copy(es.series[sel][j].T)
    #         series_j[~template.T] = 0
    #         series_k = copy(es.series[sel][k].T)
    #         series_k[~template.T] = 0
    #         dists[j, k] = l1_dist(series_j.flatten(), series_k.flatten(), es.nb_events)

    # pattern prototype based on highest ll
    # ll = np.einsum("ijk,kj", es.series, es.windowed_counts*template.T)
    id_max = np.argmax(ll)
    # id_max = np.argmin(np.sum(np.sort(dists,1)[:,:50], 1))  ########
    prototype = copy(es.series[id_max].T)
    prototype[~template.T] = 0
    plt.figure()
    plt.pcolormesh(prototype)

    # distances to prototype
    dists = np.ones(n) * np.inf
    for j in range(n):
        series_j = copy(es.series[j].T)
        series_j[~template.T] = 0
        dists[j] = cross_product_distance(series_j.flatten(), prototype.flatten(), es.nb_events)
    # plt.figure()
    # plt.scatter(range(len(dists)), dists)

    # cluster of prototype
    plt.figure()
    sel = dists < np.quantile(dists, 0.05)
    plt.pcolormesh(np.sum(es.series[sel], 0).T)

    # end loop
    select_outer[select_outer] = ~sel

plt.figure()
es = EventSeries().from_setlistfiles(fn_tos, window=3, intonly=True,
                                     selected=select_outer,
                                     constraints=ApneaConstraints,
                                     max_series_length=20)
plt.pcolormesh(np.sum(es.series, 0).T)
plt.show()

print(1)
