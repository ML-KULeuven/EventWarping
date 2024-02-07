import ast
import os
import pickle
import sys

sys.path += ["C:\\users\\dries\\eventwarping2\\src"]

from pathlib import Path

from eventwarping.formats import setlistfile2setlistsfile
from eventwarping.eventseries import EventSeries
from eventwarping.constraints import MaxMergeSymbolConstraint, \
    NoMergeTooDistantSymbolSetConstraint, NoXorMergeSymbolSetConstraint
import matplotlib.pyplot as plt
import numpy as np

from eventwarping.window import LinearScalingWindow


def cross_product_distance(s, t):
    return - np.sum(s * t) - 5 * np.sum(s[:5] * t[:5])   #################


def l1_dist(s, t):
    return - np.sum(s == t) - 5 * np.sum(s[:5] == t[:5])   #################


def dist_respiratory_itemsets(a, b, d_empty=0, n_categories=10):
    # sum of the maximal distance in peak heights per signal between itemsets a and b
    if a is None:
        return 0
    dist = 0
    for k in range(4):  # signals
        l = 5 + n_categories * k  # first of signal
        items_in_signal = set(range(l, l + n_categories))
        a_signal = a.intersection(items_in_signal)
        b_signal = b.intersection(items_in_signal)
        if len(a_signal) > 0 and len(b_signal) > 0:
            dist += np.max([np.abs(m - n)
                            for m in a_signal
                            for n in b_signal], initial=0)
        elif len(a_signal) > 0 or len(b_signal) > 0:
            dist += d_empty
    return dist


def dist_respiratory_series(s, t, d_empty=0, n_categories=10):
    d = 0
    for j in range(s.shape[1]):
        d += dist_respiratory_itemsets(set(np.argwhere(s[:, j])[:,0]), set(np.argwhere(t[:, j])[:,0]), d_empty, n_categories)
    d += 5 * np.sum(s[:5] != t[:5]) # before resp
    i = 5 + 10 * n_categories
    d += np.sum(s[i:] != t[i:])  # after resp
    return d


def make_test_set(paths, folder, events_per_sequence=25):
    " Test data. Label if start in first 10 events of the series"
    data = []
    labels = []
    for path in paths:
        if type(path) is str:
            path = Path(path)
        with path.open("r") as fp:
            data_file = eval(fp.read())
        for i in range(0, len(data_file), events_per_sequence):
            data_sequence = data_file[i:i + events_per_sequence]
            if 1 in set.union(*data_sequence[:10]):
                data += [data_sequence]
                labels += [1]
            elif 3 in set.union(*data_sequence[:10]):
                data += [data_sequence]
                labels += [2]
            elif np.random.default_rng().random() < 0.1:  # decrease random
                data += [data_sequence]
                labels += [0]
    with (folder/'test_data.txt').open("w") as fp:
        for line in data:
            fp.write(repr(line) + "\n")
    with (folder/'test_data_labels.txt').open("w") as fp:
        fp.write(repr(labels))


# folder where apnea data is stored
folder = Path(r"C:\Users\dries\input_data_pattern_mining10")
fns = [i for i in os.listdir(folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))]
fns = [folder / i for i in fns]

make_test_set(fns[1::2], folder)
fns = fns[::2]


fn_tos = []
for fn in fns:
    try:
        print(f"Path exists = {fn.exists()}: {fn}")
        fn_to = fn.parent / (fn.stem + "_series.txt")
        print(f"Saving to {fn_to}")
        fn_tos += [fn_to]
        setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=5)
    except:
        continue

# # save random series without apnea
# random_series = []
# n_series_per_file = 50
# for fn in fns:
#     if type(fn) is str:
#         fn = Path(fn)
#     with fn.open("r") as fp:
#         data = fp.read()
#     data = ast.literal_eval(data)
#     startind = np.random.randint(0, len(data)-25, n_series_per_file)
#     for i in startind:
#         random_series += [[data[i+j] for j in range(25)]]
# with Path(folder / "random.txt").open("w") as fp:
#     for series in random_series:
#         fp.write(repr(series) + "\n")
#     setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=5)
# fn_tos += [Path(folder / "random.txt")]

constraints = [
    NoMergeTooDistantSymbolSetConstraint(dist_respiratory_itemsets, 10),
    MaxMergeSymbolConstraint(5),
    NoXorMergeSymbolSetConstraint([1, 2, 3, 4])
]

param_dict = {"fns": fn_tos, "window": LinearScalingWindow(5), "intonly": True, "constraints": constraints, "max_series_length": 25}
n_iter = 10

# initialize eventseries
es_all = EventSeries().from_setlistfiles(**param_dict)
is_apnea = es_all.series[:,:,1].sum(axis=1).astype(bool)
is_hypopnea = es_all.series[:,:,3].sum(axis=1).astype(bool)
is_random = ~is_hypopnea & ~is_apnea
es_apnea = EventSeries().from_setlistfiles(selected=is_apnea, **param_dict)
es_hypopnea = EventSeries().from_setlistfiles(selected=is_hypopnea, **param_dict)

for symbol in [1, 2, 3, 4]:
    es_apnea.rescale_weights[symbol] = 5
    es_hypopnea.rescale_weights[symbol] = 5

# remove other apnea/hypopnea such that only 1 remains per series
es_apnea.series[:, :, 2][np.cumsum(es_apnea.series[:, :, 1], 1) == 0] = 0
es_apnea.series[:, :, 4][np.cumsum(es_apnea.series[:, :, 3], 1) == 0] = 0
es_apnea.series[:, ::-1, 1][np.cumsum(es_apnea.series[:, ::-1, 2], 1) == 0] = 0
es_apnea.series[:, ::-1, 3][np.cumsum(es_apnea.series[:, ::-1, 4], 1) == 0] = 0
# remove other apnea/hypopnea such that only 1 remains per series
es_hypopnea.series[:, :, 2][np.cumsum(es_hypopnea.series[:, :, 1], 1) == 0] = 0
es_hypopnea.series[:, :, 4][np.cumsum(es_hypopnea.series[:, :, 3], 1) == 0] = 0
es_hypopnea.series[:, ::-1, 1][np.cumsum(es_hypopnea.series[:, ::-1, 2], 1) == 0] = 0
es_hypopnea.series[:, ::-1, 3][np.cumsum(es_hypopnea.series[:, ::-1, 4], 1) == 0] = 0

# do alignment
for i, ws in enumerate(es_apnea.warp_yield(iterations=n_iter, restart=True)):
    print(f"=== {i + 1:>2} ===")
    es_apnea.plot_directions()
    es_apnea.plot_symbols()
for i, ws in enumerate(es_hypopnea.warp_yield(iterations=n_iter, restart=True)):
    print(f"=== {i + 1:>2} ===")

with open("es_apnea.pkl", 'wb') as file:
    pickle.dump(es_apnea, file)
with open("es_hypopnea.pkl", 'wb') as file:
    pickle.dump(es_hypopnea, file)
with open("es_apnea.pkl", 'rb') as file:
    es_apnea = pickle.load(file)
with open("es_hypopnea.pkl", 'rb') as file:
    es_hypopnea = pickle.load(file)

# select events in which the pattern in visible
items_per_event = es_apnea.warped_series.sum(axis=(0,2))
selected_events_apnea = items_per_event > (np.max(items_per_event)/100)
items_per_event = es_hypopnea.warped_series.sum(axis=(0,2))
selected_events_hypopnea = items_per_event > (np.max(items_per_event)/100)

# define new event_series to calculate the likelihood on
param_dict['fns'] = [folder/'test_data.txt']
es_apnea_model = EventSeries().from_setlistfiles(**param_dict)
es_apnea_model.series[:, :, :4] = 0
es_hypopnea_model = EventSeries().from_setlistfiles(**param_dict)
es_hypopnea_model.series[:, :, :4] = 0

es_apnea_model.warp_with_model(model=es_apnea, iterations=n_iter)
ll_apnea = es_apnea_model.likelihood(model=es_apnea, laplace_smoothing=0.1, exclude_items=(1, 2, 3, 4), selected_events=selected_events_apnea)
es_hypopnea_model.warp_with_model(model=es_hypopnea, iterations=n_iter)
ll_hypopnea = es_hypopnea_model.likelihood(model=es_hypopnea, laplace_smoothing=0.1, exclude_items=(1, 2, 3, 4), selected_events=selected_events_hypopnea)

with open("es_apnea_model.pkl", 'wb') as file:
    pickle.dump(es_apnea_model, file)
with open("es_hypopnea_model.pkl", 'wb') as file:
    pickle.dump(es_hypopnea_model, file)

with (folder/'test_data_labels.txt').open('r') as file:
    labels_test = np.array(eval(file.read()))
is_apnea = labels_test == 1
is_hypopnea = labels_test == 2
is_random = labels_test == 0

es_apnea_model.plot_symbols(filter_series=is_apnea)
es_hypopnea_model.plot_symbols(filter_series=is_hypopnea)

plt.figure()
plt.hist((ll_apnea)[is_apnea],200)
plt.hist((ll_apnea)[is_hypopnea],200)
plt.hist((ll_apnea)[is_random],200)

plt.figure()
plt.hist((ll_hypopnea)[is_apnea],200)
plt.hist((ll_hypopnea)[is_hypopnea],200)
plt.hist((ll_hypopnea)[is_random],200)

plt.figure()
plt.hist((ll_apnea/ll_hypopnea)[is_apnea],200)
plt.hist((ll_apnea/ll_hypopnea)[is_hypopnea],200)
plt.hist((ll_apnea/ll_hypopnea)[is_random],200)


n_events = np.sum(np.sign(es_apnea_model._warped_series), axis=(1,2)).astype(float)
n_events += np.random.random(len(n_events)) - 0.5

plt.figure()
plt.scatter(n_events[is_apnea], ll_apnea[is_apnea], label='is_apnea')
plt.scatter(n_events[is_hypopnea], ll_apnea[is_hypopnea], label='is_hypopnea')
plt.scatter(n_events[is_random], ll_apnea[is_random], label='is_random')
plt.xlabel('n_events')
plt.title('apnea_model')
plt.ylabel('log_likelihood')

plt.figure()
plt.scatter(n_events[is_apnea], ll_apnea[is_apnea], label='is_apnea')
plt.scatter(n_events[is_hypopnea], ll_apnea[is_hypopnea], label='is_hypopnea')
plt.scatter(n_events[is_random], ll_apnea[is_random], label='is_random')
plt.xlabel('n_events')
plt.title('hypopnea_model')
plt.ylabel('log_likelihood')

plt.figure()
plt.hist(n_events[is_apnea],200)
plt.hist(n_events[is_hypopnea],200)
plt.hist(n_events[is_random],200)

plt.show()

x = es_apnea_model.warped_series.reshape(5903, 25*50)
y = labels_test
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix as CM
dt = DTC(min_samples_leaf=20)
y_pred = dt.fit(x[::2], y[::2]).predict(x[1::2])
print(CM(y_pred, y[1::2]))
a = dt.feature_importances_.reshape(25,50)
plt.imshow(a.T)
print(1)
