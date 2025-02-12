import ast
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path += [r"C:\Users\dries\Python projects\AItoolkit"]
from AT.blocks.data.data_set import BaseSet
from pathlib import Path
from eventwarping.formats import smooth_series



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