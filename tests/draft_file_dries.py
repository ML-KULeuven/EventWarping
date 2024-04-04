import ast
import os
import pickle
import sys

sys.path += ["C:\\users\\dries\\eventwarping2\\src"]

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix as CM, precision_score, \
    recall_score
from copy import copy
from sklearn.tree import plot_tree

from eventwarping.formats import setlistfile2setlistsfile, smooth_series
from eventwarping.eventseries import EventSeries
import matplotlib.pyplot as plt
import numpy as np


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


def make_test_set(paths, folder, events_per_sequence=25, symbol_ordenings=None):
    " Test data. Label apnea/hypopnea if start in first 10 events of the series"
    data = []
    labels = []
    data_origin = []
    rng = np.random.default_rng(seed=0)
    for path in paths:
        if type(path) is str:
            path = Path(path)
        with path.open("r") as fp:
            data_file = ast.literal_eval(fp.read())

        data_file = smooth_series([data_file], symbol_ordenings)[0]

        for i in range(0, len(data_file), events_per_sequence):
            data_sequence = data_file[i:i + events_per_sequence]
            if 1 in set.union(*data_sequence[:10]):
                data += [data_sequence]
                labels += [1]
            elif 3 in set.union(*data_sequence[:10]):
                data += [data_sequence]
                labels += [2]
            elif rng.random() < 0.1:  # decrease random
                data += [data_sequence]
                labels += [0]
            else:
                continue
            data_origin += [(path.__str__(), i)]
    with (folder/'test_data.txt').open("w") as fp:
        for line in data:
            fp.write(repr(line) + "\n")
    with (folder/'test_data_labels.txt').open("w") as fp:
        fp.write(repr(labels))
    with (folder/'test_data_origins.txt').open("w") as fp:
        fp.write(repr(data_origin))


def train_event_series_alignment(fns, param_dict, n_iter, folder, file_name_apnea="es_apnea", file_name_hypopnea="es_hypopnea", excluded_events=None):
    # initialize eventseries
    es_all = EventSeries().from_setlistfiles(fns=fns, **param_dict)
    is_apnea = es_all.series[:,:,1].sum(axis=1).astype(bool)
    is_hypopnea = es_all.series[:,:,3].sum(axis=1).astype(bool)
    es_apnea = EventSeries().from_setlistfiles(selected=is_apnea, fns=fns, **param_dict)
    es_hypopnea = EventSeries().from_setlistfiles(selected=is_hypopnea, fns=fns, **param_dict)

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

    # set excluded events to 0 in series
    for event in excluded_events:
        es_apnea.series[:, :, event] = 0
        es_hypopnea.series[:, :, event] = 0

    # do alignment
    for i, ws in enumerate(es_apnea.warp_yield(iterations=n_iter, restart=True)):
        print(f"=== {i + 1:>2} ===")
    for i, ws in enumerate(es_hypopnea.warp_yield(iterations=n_iter, restart=True)):
        print(f"=== {i + 1:>2} ===")

    with open(folder / (file_name_apnea + ".pkl"), 'wb') as file:
        pickle.dump(es_apnea, file)
    with open(folder / (file_name_hypopnea + ".pkl"), 'wb') as file:
        pickle.dump(es_hypopnea, file)

    return es_apnea, es_hypopnea


def warp_test_data_with_model(files_test_data, param_dict, n_iter, es_apnea, es_hypopnea, file_name_apnea_model="es_apnea_model", file_name_hypopnea_model="es_hypopnea_model", excluded_events=None):
    es_apnea_model = EventSeries().from_setlistfiles(files_test_data,
                                                     **param_dict)
    es_apnea_model.series[:, :, :5] = 0
    es_hypopnea_model = EventSeries().from_setlistfiles(files_test_data,
                                                        **param_dict)
    es_hypopnea_model.series[:, :, :5] = 0

    # set excluded events to 0 in series
    for event in excluded_events:
        es_apnea.series[:, :, event] = 0
        es_hypopnea.series[:, :, event] = 0

    es_apnea_model.warp_with_model(model=es_apnea, iterations=n_iter)
    es_hypopnea_model.warp_with_model(model=es_hypopnea, iterations=n_iter)

    with open(results_folder / (file_name_apnea_model + ".pkl"), 'wb') as file:
        pickle.dump(es_apnea_model, file)
    with open(results_folder / (file_name_hypopnea_model + ".pkl"), 'wb') as file:
        pickle.dump(es_hypopnea_model, file)

    return es_apnea_model, es_hypopnea_model


def do_evaluations(labels, model, model_name, model_class):
    is_apnea = labels == 1
    is_hypopnea = labels == 2
    is_random = labels == 0
    is_wake = np.sum(model.series[:,:,48],1) > 5
    labels_sleep = labels[~is_wake]

    if model_class == 1:
        is_class = is_apnea[~is_wake]
        class_name = 'apnea'
    if model_class == 2:
        is_class = is_hypopnea[~is_wake]
        class_name = 'hypopnea'

    # plot symbols of model
    model.plot_symbols(filter_series=is_apnea, title=f'{class_name} data warped by {model_name} model')

    # learn classifiers on warped series
    x = np.sign(model.warped_series[~is_wake].reshape(-1, 25 * 50))
    x_nowarp = np.sign(model.series[~is_wake].reshape(-1, 25 * 50))
    y = labels_sleep

    def rf_clf(x_train, x_test, y_train, y_test):
        clf = RFC(min_samples_leaf=50, max_features=100, n_estimators=100)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        cm = CM(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=None)
        rec = recall_score(y_test, y_pred, average=None)
        print(cm, prec, rec)
        return cm, prec, rec

    print('before warping')
    cm, prec, rec = rf_clf(x_nowarp[::2], x_nowarp[1::2], y[0::2], y[1::2])

    print('after warping')
    cm, prec, rec = rf_clf(x[::2], x[1::2], y[0::2], y[1::2])

    print('after warping train set')
    cm, prec, rec = rf_clf(x[::2], x[::2], y[0::2], y[::2])

    print(f'after warping: 2 classes ({model_name} or not)')
    cm, prec, rec = rf_clf(x[::2], x[1::2], is_class[::2], is_class[1::2])

    # decision tree plot
    with (items_folder / 'alphabet.pkl').open('rb') as file:
        alphabet = pickle.load(file)
    alphabet[0] = ""
    # select some instances
    x7 = copy(x).reshape((-1, 25, 50))
    for i in range(len(x7)):
        x7[i, :, 45] = (sum(x7[i, :, 45]) > 5) * 1
    x7 = x7.reshape((-1, 25 * 50))
    sel = np.array([True] * len(x7))
    # learn and plot tree (2 classes)
    dt = DTC(min_samples_leaf=30)
    dt.fit(x7[sel][::2], is_class[sel][::2])
    plt.figure()
    plot_tree(dt,
              feature_names=[(i // 50, alphabet[i % 50]) for i in range(1250)],
              fontsize=5, filled=True, impurity=False)
    # learn and plot tree (3 classes)
    dt = DTC(min_samples_leaf=30)
    dt.fit(x7[sel][::2], y[::2])
    plt.figure()
    plot_tree(dt,
              feature_names=[(i // 50, alphabet[i % 50]) for i in range(1250)],
              fontsize=5, filled=True, impurity=False)

    # plot decrease of 100 apnea/hypopnea
    fig, axs = plt.subplots(10, 10)
    fig.suptitle(f'{model_name} data after warping with {model_name} model')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    j = 0
    for i in range(len(x)):
        if y[i] == model_class:  # hypopnea
            axs[j // 10, j % 10].imshow(x[i].reshape((25, 50))[0:10, 5:26].T,
                                        aspect='auto')
            axs[j // 10, j % 10].set_xticklabels([])
            axs[j // 10, j % 10].set_yticklabels([])
            j += 1
        if j == 100:
            break
    fig, axs = plt.subplots(10, 10)
    fig.suptitle(f'{model_name} data before warping')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    j = 0
    for i in range(len(x_nowarp)):
        if y[i] == model_class:
            axs[j // 10, j % 10].imshow(x_nowarp[i].reshape((25, 50))[0:10, 5:26].T,
                                        aspect='auto')
            axs[j // 10, j % 10].set_xticklabels([])
            axs[j // 10, j % 10].set_yticklabels([])
            j += 1
        if j == 100:
            break

    # # influence of start position apnea/hypopnea on accuracy
    # test_data = read_test_data(results_folder)[~is_wake][1::2]
    # apnea_index_per_test_series = [np.argwhere([len(i.intersection({1, 2, 3, 4})) > 0 for i in j]) for j in test_data]
    # apnea_indices = np.array([apnea_index_per_test_series[i][0] for i in range(len(test_data)) if is_apnea[~is_wake][1::2][i]])[:,0]
    # hypopnea_indices = np.array([apnea_index_per_test_series[i][0] for i in range(len(test_data)) if is_hypopnea[~is_wake][1::2][i]])[:, 0]
    # for i in range(10):
    #     a = (y_pred == y[1::2])[[~is_wake][1::2]][apnea_indices == i]
    #     print(i, len(a), np.mean(a))
    # for i in range(10):
    #     a = (y_pred == y[1::2])[is_hypopnea[~is_wake][1::2]][hypopnea_indices == i]
    #     print(i, len(a), np.mean(a))

    plt.show()
    print('evaluation ended')


def read_test_data(folder):
    test_data = []
    with (folder / 'test_data.txt').open('r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            test_data += [eval(line)]
    return test_data


def series_to_diff_series(series, symbol_ordenings):
    nb_series, nb_events, nb_items = series.shape
    diff_series = np.zeros((nb_series, nb_events, len(symbol_ordenings)))
    for k, ordening in enumerate(symbol_ordenings):
        series_ordening = series[:,:,ordening]
        for j, serie in enumerate(series_ordening):
            ind = np.argwhere(serie)
            serie_max = np.array([np.max(ind[ind[:, 0] == i, 1], initial=-1000) for i in range(nb_events)])
            serie_min = np.array([np.min(ind[ind[:, 0] == i, 1], initial=1000) for i in range(nb_events)])
            serie_pos_diff = np.array([serie_min[i] - serie_max[i - 1] for i in range(1, nb_events)])
            serie_neg_diff = np.array([serie_max[i] - serie_min[i - 1] for i in range(1, nb_events)])
            sel = np.array([serie_pos_diff[i] * serie_neg_diff[i] >= 0 for i in range(nb_events-1)])
            sel_pos = np.abs(serie_pos_diff) > np.abs(serie_neg_diff)
            series_diff = np.zeros(nb_events)
            series_diff[1:][sel_pos * sel] = serie_pos_diff[sel_pos * sel]
            series_diff[1:][~sel_pos * sel] = serie_neg_diff[~sel_pos * sel]
            diff_series[j,:,k] = series_diff
    return diff_series


# define data and parameters
symbol_ordenings = [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]

items_folder = Path(r"C:\Users\dries\input_data_pattern_mining11")
results_folder = Path(r"C:\Users\dries\input_data_pattern_mining11_smoothed")
fns = [i for i in os.listdir(items_folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))]
fns = [items_folder / i for i in fns]
# make_test_set(fns[1::2], results_folder, symbol_ordenings=symbol_ordenings)
fns = fns[::2]

fn_tos = []
for fn in fns:
    print(f"Path exists = {fn.exists()}: {fn}")
    fn_to = results_folder / (fn.stem + "_series.txt")
    print(f"Saving to {fn_to}")
    fn_tos += [fn_to]
    # setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=5, symbol_ordenings=symbol_ordenings)


# for max_dist in [10]:
#     print(f"******************* {max_dist} ********************")
#     file_name_apnea = "es_apnea" + f"max_{max_dist}"
#     file_name_hypopnea = "es_hypopnea" + f"max_{max_dist}"
#     file_name_apnea_model = "es_apnea_model" + f"max_{max_dist}"
#     file_name_hypopnea_model = "es_hypopnea_model" + f"max_{max_dist}"
#
#     constraints = [
#         NoMergeTooDistantSymbolSetConstraint(dist_respiratory_itemsets, max_dist),
#         MaxMergeSymbolConstraint(5),
#         NoXorMergeSymbolSetConstraint([1, 2, 3, 4])
#     ]
#     param_dict = {"window": LinearScalingWindow(5), "intonly": True, "constraints": constraints, "max_series_length": 25}
#     n_iter = 10
#
#     # train event alignment
#     es_apnea, es_hypopnea = train_event_series_alignment(fn_tos, param_dict, n_iter, results_folder, file_name_apnea, file_name_hypopnea)
#
#     with open(results_folder / (file_name_apnea + '.pkl'), 'rb') as file:
#         es_apnea = pickle.load(file)
#     with open(results_folder / (file_name_hypopnea + '.pkl'), 'rb') as file:
#         es_hypopnea = pickle.load(file)
#
#     # warp test data with model
#     files_test_data = [results_folder / 'test_data.txt']
#     es_apnea_model, es_hypopnea_model = warp_test_data_with_model(files_test_data, param_dict, n_iter, es_apnea, es_hypopnea, file_name_apnea_model, file_name_hypopnea_model)


for max_dist in [10]:
    print(f"******************* {max_dist} ********************")
    file_name_apnea_model = "es_apnea_model" + f"max_{max_dist}.pkl"
    file_name_hypopnea_model = "es_hypopnea_model" + f"max_{max_dist}.pkl"

    with open(results_folder / file_name_apnea_model, 'rb') as file:
        es_apnea_model = pickle.load(file)
    with open(results_folder / file_name_hypopnea_model, 'rb') as file:
        es_hypopnea_model = pickle.load(file)

    # diff_series = series_to_diff_series(es_apnea_model.warped_series, symbol_ordenings)
    # es_apnea_model.diff_series = diff_series

    # do different evaluations
    with (results_folder / 'test_data_labels.txt').open('r') as file:
        labels_test = np.array(eval(file.read()))
    print("----------apnea-------------")
    do_evaluations(labels_test, es_apnea_model, 'apnea', 1)
    # print("--------hypopnea------------")
    # do_evaluations(labels_test, es_hypopnea_model, 'hypopnea', 2)

plt.show()

print(1)
