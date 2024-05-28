import ast
import os
import pickle
import sys
from copy import copy

from src.eventwarping.constraints import NoMergeTooDistantSymbolSetConstraint, MaxMergeSymbolConstraint, \
    NoXorMergeSymbolSetConstraint
from src.eventwarping.window import LinearScalingWindow
from tests.apnea_evaluation import do_evaluations
from AT.respiratory.pattern_mining.plot import plot_itemsets
from AT.blocks.data.data_set import BaseSet
from AT.respiratory import uza_multilabeled_single_files

sys.path += [r"C:\Users\dries\Python projects\EventWarping\src"]
sys.path += [r"C:\Users\dries\Python projects\AItoolkit"]

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier as RFC
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
    "Test data. Label apnea/hypopnea if start in first 10 events of the series"
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
    if excluded_events is not None:
        for event in excluded_events:
            es_apnea.series[:, :, event] = 0
            es_hypopnea.series[:, :, event] = 0

    # do alignment
    for i, ws in enumerate(es_apnea.warp_yield(iterations=n_iter, restart=True)):
        print(f"=== {i + 1:>2} ===")
    for i, ws in enumerate(es_hypopnea.warp_yield(iterations=n_iter, restart=True)):
        print(f"=== {i + 1:>2} ===")

    with open(folder / (file_name_apnea), 'wb') as file:
        pickle.dump(es_apnea, file)
    with open(folder / (file_name_hypopnea), 'wb') as file:
        pickle.dump(es_hypopnea, file)

    return es_apnea, es_hypopnea


def warp_test_data_with_model(files_test_data, param_dict, n_iter, es_apnea, es_hypopnea, models_folder, file_name_apnea_model="es_apnea_model", file_name_hypopnea_model="es_hypopnea_model", excluded_events=None):
    es_apnea_model = EventSeries().from_setlistfiles(files_test_data,
                                                     **param_dict)
    es_apnea_model.series[:, :, :5] = 0
    es_hypopnea_model = EventSeries().from_setlistfiles(files_test_data,
                                                        **param_dict)
    es_hypopnea_model.series[:, :, :5] = 0

    # set excluded events to 0 in series
    if excluded_events is not None:
        for event in excluded_events:
            es_apnea.series[:, :, event] = 0
            es_hypopnea.series[:, :, event] = 0

    es_apnea_model.warp_with_model(model=es_apnea, iterations=n_iter)
    es_hypopnea_model.warp_with_model(model=es_hypopnea, iterations=n_iter)

    with open(models_folder / (file_name_apnea_model), 'wb') as file:
        pickle.dump(es_apnea_model, file)
    with open(models_folder / (file_name_hypopnea_model), 'wb') as file:
        pickle.dump(es_hypopnea_model, file)

    return es_apnea_model, es_hypopnea_model


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
            diff_series[j, :, k] = series_diff
    return diff_series


def predict_apnea_per_file(filename, clf, model, brt_files_apnea=None):
    """Filename of itemsets, brt_files_apnea: files to extract apnea from, otherwise from itemsets"""
    # read data
    with open(filename, 'r') as f:
        setlist = eval(f.read())
    with open(str(filename).replace('itemsets.txt', 'itemset_times.txt'), 'r') as f:
        times = eval(f.read())
    setlist = smooth_series([setlist], symbol_ordenings)[0]

    # transform sets to matrix of series
    shift = 5
    window_size = 25
    data_matrix = np.zeros((len(setlist), 50))
    for i, s in enumerate(setlist):
        for j in s:
            data_matrix[i, j] = 1
    data_windowed = np.zeros((len(data_matrix) // shift - (window_size // shift), window_size, 50))
    for i in range(len(data_windowed)):
        data_windowed[i] = data_matrix[i * shift:(i * shift + window_size)]

    # warp series with model
    data_aligned = model.align_series_times(data_windowed, iterations=10)[0]

    # plot prediction of clf
    y_pred = clf.predict_proba(data_aligned.reshape((len(data_windowed), window_size * 50)))
    plt.figure()
    plt.plot(times[::shift][:len(y_pred)], y_pred)

    # add apnea and hypopnea to plot
    from AT.respiratory.itemset_construction.items_of_psg import _read_respiratory_events
    if brt_files_apnea:
        # multilabeled, so plot apnea of each file
        for j, path_brt in enumerate(brt_files_apnea):
            starts, ends, is_hypopnea = _read_respiratory_events(path_brt)
            plt.hlines([1.05 + 0.05*j] * sum(~is_hypopnea), starts[~is_hypopnea], ends[~is_hypopnea], color='k')
            plt.hlines([1.05 + 0.05*j] * sum(is_hypopnea), starts[is_hypopnea], ends[is_hypopnea], color='b')
    else:
        # apnea based on itemsets
        is_apnea = np.array([1 in i for i in setlist])
        is_hypopnea = np.array([3 in i for i in setlist])
        plt.hlines([1.05] * sum(is_apnea), np.array(times)[is_apnea] - 15, np.array(times)[is_apnea] + 15, color='k')
        plt.hlines([1.05] * sum(is_hypopnea), np.array(times)[is_hypopnea] - 15, np.array(times)[is_hypopnea] + 15, color='b')

        path_brt = [i for i in BaseSet.TRAIN|BaseSet.VAL if filename.parts[-1].replace('_itemsets.txt', '') in i][0]

    # add feature plots
    d = np.repeat(data_matrix, np.diff(np.array([0] + times) // 0.5).astype(int),
                  axis=0).T  # data aligned over time
    plt.imshow(d, extent=(0, max(times), -1, 0), aspect='auto')
    plt.ylim((-1, 1.3))

    # plot itemsets
    # if path_brt.startswith('\\\\OSG-110'):
    #     brt_file = path_brt.replace(r'\\OSG-110\USB_Full',
    #                                 r'\\osgnet\signals\OSG-110')
    # else:
    #     brt_file = path_brt.replace(r'\\osgnet\data', r'\\osgnet\signals')
    # dir_itemsets = str(filename.parents[0]) + "//"
    # plot_itemsets(str(brt_file), dir_itemsets)
    return y_pred


if __name__ == '__main__':
    # define data and parameters
    symbol_ordenings = [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]

    items_folder = Path(r"C:\Users\dries\python projects\itemsets_new")  # training and files test set
    items_smooth_folder = Path(r"C:\Users\dries\python projects\smoothed_itemsets_new")
    models_folder = Path(r"C:\Users\dries\python projects\smoothed_itemsets_new")
    fns = [i for i in os.listdir(items_folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))]
    fns = [items_folder / i for i in fns]
    fns_test = copy(fns[1::2])
    fns = fns[::2]  # train set

    max_dist = 10
    file_name_apnea = "es_apnea" + f"max_{max_dist}.pkl"
    file_name_hypopnea = "es_hypopnea" + f"max_{max_dist}.pkl"
    file_name_apnea_model = "es_apnea_model" + f"max_{max_dist}.pkl"
    file_name_hypopnea_model = "es_hypopnea_model" + f"max_{max_dist}.pkl"

    constraints = [
        NoMergeTooDistantSymbolSetConstraint(dist_respiratory_itemsets, max_dist),
        MaxMergeSymbolConstraint(5),
        NoXorMergeSymbolSetConstraint([1, 2, 3, 4])
    ]
    param_dict = {"window": LinearScalingWindow(5), "intonly": True, "constraints": constraints, "max_series_length": 25}
    n_iter = 10

    ##############################################

    # # save smoothed itemsets
    # fn_tos = []
    # for fn in fns:
    #     print(f"Path exists = {fn.exists()}: {fn}")
    #     fn_to = items_smooth_folder / (fn.stem + "_series.txt")
    #     print(f"Saving to {fn_to}")
    #     fn_tos += [fn_to]
    #     setlistfile2setlistsfile(fn, fn_to, start={1,3}, stop={2,4}, margin=5, symbol_ordenings=symbol_ordenings)

    ##############################################

    # # train event alignment
    # # train_event_series_alignment(fn_tos, param_dict, n_iter, models_folder, file_name_apnea, file_name_hypopnea)
    # with open(models_folder / (file_name_apnea), 'rb') as file:
    #     es_apnea = pickle.load(file)
    # with open(models_folder / (file_name_hypopnea), 'rb') as file:
    #     es_hypopnea = pickle.load(file)
    #
    # # warp test data with model
    # make_test_set(fns_test, items_smooth_folder, symbol_ordenings=symbol_ordenings)
    # files_test_data = [items_smooth_folder / 'test_data.txt']
    # # warp_test_data_with_model(files_test_data, param_dict, n_iter, es_apnea, es_hypopnea, models_folder, file_name_apnea_model, file_name_hypopnea_model)
    # with open(models_folder / file_name_apnea_model, 'rb') as file:
    #     es_apnea_model = pickle.load(file)
    # with open(models_folder / file_name_hypopnea_model, 'rb') as file:
    #     es_hypopnea_model = pickle.load(file)

    ##############################################

    # # evaluation on test data set
    # diff_series = series_to_diff_series(es_apnea_model.warped_series, symbol_ordenings)
    # es_apnea_model.diff_series = diff_series

    # with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
    #     labels_test = np.array(eval(file.read()))
    # print("----------apnea model-------------")
    # do_evaluations(labels_test, es_apnea_model, 'apnea', 1, items_folder, items_smooth_folder)
    # plt.show()

    ##############################################

    # evaluation on individual file
    ## train clf on test set data
    with open(models_folder / file_name_apnea_model, 'rb') as file:
        model = pickle.load(file)
    x = np.sign(model.warped_series)
    x = x.reshape((len(x), -1))
    with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
        y = np.array(eval(file.read()))
    clf = RFC(min_samples_leaf=30, max_features=100, n_estimators=100)
    clf.fit(x, y)

    ## evaluate on file
    folder_test_files = Path(r"C:\Users\dries\python projects\itemsets_multilabeled")
    files = [folder_test_files / i for i in os.listdir(folder_test_files) if (i.endswith('itemsets.txt'))]
    for i in files[1::5]:
        brt_file = [j for j in uza_multilabeled_single_files if str(i.parts[-1]).replace('_itemsets.txt','') in j][0]
        pred = predict_apnea_per_file(i, clf, model, uza_multilabeled_single_files[brt_file])

    print(1)
