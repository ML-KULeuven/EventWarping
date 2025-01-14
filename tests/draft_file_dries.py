import ast
import os
import pickle
import sys
from copy import copy

from scipy.interpolate import make_interp_spline

from src.eventwarping.constraints import NoMergeTooDistantSymbolSetConstraint, MaxMergeSymbolConstraint, \
    NoXorMergeSymbolSetConstraint
from src.eventwarping.window import LinearScalingWindow
from tests.apnea_evaluation import do_evaluations
from AT.respiratory.itemset_construction.plot import plot_itemsets
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

    # with open(folder / (file_name_apnea), 'wb') as file:
    #     pickle.dump(es_apnea, file)
    # with open(folder / (file_name_hypopnea), 'wb') as file:
    #     pickle.dump(es_hypopnea, file)

    return es_apnea, es_hypopnea


def train_event_series_alignment_no_apnea(test_data_files, labels_test_data, param_dict, n_iter, folder, file_name="es_none", excluded_events=None):
    """
    Train warping on 10000 sequences without apnea or hypopnea
    """
    # initialize eventseries
    sel = labels_test_data == 0
    sel *= np.cumsum(sel) <= 10000
    es_none = EventSeries().from_setlistfiles(selected=labels_test_data==0, fns=test_data_files, **param_dict)

    # set excluded events to 0 in series
    if excluded_events is not None:
        for event in excluded_events:
            es_apnea.series[:, :, event] = 0
            es_hypopnea.series[:, :, event] = 0

    # do alignment
    for i, ws in enumerate(es_none.warp_yield(iterations=n_iter, restart=True)):
        print(f"=== {i + 1:>2} ===")

    # with open(folder / (file_name), 'wb') as file:
    #     pickle.dump(es_none, file)

    return es_none


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

    # with open(models_folder / (file_name_apnea_model), 'wb') as file:
    #     pickle.dump(es_apnea_model, file)
    # with open(models_folder / (file_name_hypopnea_model), 'wb') as file:
    #     pickle.dump(es_hypopnea_model, file)

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

def add_grid(ax):
    ax.grid(axis='y', linewidth=2, color='k')
    for x in range(27):
        ax.axvline(x-0.5, color='gray', linewidth=0.5)
    for y in range(48):
        ax.axhline(y - 1.5, color='gray', linewidth=0.5)
    return ax

def plot_warping(model, labels, label, name='plot1'):
    """
    Plot the counts of all symbols over all events (aggregated over the series) before and after warping.
    """
    ticks = np.array([1.5,11.5,21.5,31.5, 41.5, 45.5]) - 2
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax    ', 'other']

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,6))

    ax = axs[0]
    im = model.series[labels == label].mean(axis=0).T
    implot = ax.imshow(im[5:49], vmin=0, vmax=0.65)
    # ax = add_grid(ax)
    ax.set_xlabel('t')
    ax.set_ylabel('item')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_title('before warping')

    ax = axs[1]
    im = model.get_counts(ignore_merged=True, filter_series=labels == label) / sum(labels==label)
    implot = ax.imshow(im[5:49], vmin=0, vmax=0.65)
    # ax = add_grid(ax)
    ax.set_xlabel('t')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_title('after warping')

    fig.colorbar(implot, location='right', ax=axs, shrink=0.75)

    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

def plot_density(models, name):
    """
    Plot the density
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(models), figsize=(2*len(models)+1.5, 6))
    ticks = np.array([1.5, 11.5,21.5,31.5, 41.5, 45.5])
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax     ', 'other']

    for i, model in enumerate(models):
        ax = axs[i]
        im = model.get_counts(ignore_merged=True) / len(model.series)
        im[3:5] += im[1:3]
        implot = ax.imshow(im[3:49, :15], vmin=0, vmax=0.9)
        # fig.colorbar(implot, location='top', ax=ax)
        ax.set_xlabel('t')
        if i == 0:
            ax.set_ylabel('item')
        ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
        ax.set_ylim(-0.5, 45.5)

    fig.colorbar(implot, location='right', ax=axs, shrink=0.75)

    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)

def plot_single_series(model, name, nr, density):
    """
    Series 53 = [DZ_00001_0000414_itemsets.txt', '10500']
    """
    ticks = np.array([1.5, 11.5, 21.5, 31.5, 41.5, 45.5]) - 2
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax    ', 'other']

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    ax = axs[0]
    im = model.series[nr].T
    ax.imshow(1-im[5:49], cmap='gray')
    ax = add_grid(ax)
    ax.set_xlabel('t')
    ax.set_ylabel('item')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_xlim(-0.5, 24.5)
    ax.set_title('before warping')

    ax = axs[1]
    im = np.sign(model.warped_series[nr].T)
    ax.imshow(1-im[5:49], cmap='gray')
    ax = add_grid(ax)
    ax.set_xlabel('t')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 43.5)
    ax.set_xlim(-0.5, 24.5)
    ax.set_title('after warping')

    # ax = axs[1]
    # im = np.sign(model.warped_series[nr].T) * density
    # ax.imshow(im[5:49])
    # ax.set_xlabel('t')
    # ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    # ax.set_ylim(-0.5, 43.5)

    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)
    return fig,ax

def plot_training_examples(apnea_model, hypopnea_model, name, nr_apnea, nr_hypopnea):
    ticks = np.array([ 1.5, 11.5, 21.5, 31.5, 41.5, 45.5])
    tick_labels = ['', 'cannula   ', 'thermistor  ', 'abdomen   ', 'thorax    ', 'other']

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    ax = axs[0]
    im = apnea_model.series[nr_apnea].T
    im[3:5] += im[1:3]
    ax.imshow(1 - im[3:49], cmap='gray')
    ax.grid(axis='y', linewidth=2, color='k')
    for x in range(27):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(48):
        ax.axhline(y - 1.5, color='gray', linewidth=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('item')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 45.5)
    ax.set_xlim(-0.5, 20.5)

    ax = axs[1]
    im = hypopnea_model.series[nr_hypopnea].T
    ax.imshow(1 - im[3:49], cmap='gray')
    ax.grid(axis='y', linewidth=2, color='k')
    for x in range(27):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(48):
        ax.axhline(y - 1.5, color='gray', linewidth=0.5)
    ax.set_xlabel('t')
    ax.set_yticks(ticks, labels=tick_labels, rotation='vertical')
    ax.set_ylim(-0.5, 45.5)
    ax.set_xlim(-0.5, 20.5)

    fig.savefig(name, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_directions(filename):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fn = Path(__file__).parent / "rsrc" / "example9.txt"
    es = EventSeries.from_file(fn, window=LinearScalingWindow(3), constraints=[])

    symbol = [0, 1]
    es.compute_windowed_counts()
    es.compute_rescaled_counts()
    es.compute_warping_directions()

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey='row',
                            figsize=(10, 4))
    cnts = es.get_counts(ignore_merged=True)
    colors = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    for curidx, cursymbol in enumerate(symbol):
        curcnts = cnts[cursymbol]

        # Counts
        ax = axs[0, curidx]
        ax.set_title(f"Symbol {es.int2symbol.get(cursymbol, cursymbol)}")
        ax.bar(list(range(len(curcnts))), curcnts, color=colors[0], label="Counts")
        if curidx == 0:
            ax.legend(bbox_to_anchor=(-0.15, 1), loc='upper right')

        # Scaled counts
        ax = axs[1, curidx]
        w_dens = es.ws[:, :, curidx].sum(axis=0)
        w_dens /= w_dens.sum(axis=0)
        ax.bar(list(range(len(curcnts))), w_dens, color=colors[0], label="Weighted counts $c_{i,S}$")
        ax.plot(es._smoothed_counts[cursymbol], '-o', color=colors[3], label="Density $d_{i,S}$")

        if curidx == 0:
            ax.legend(bbox_to_anchor=(-0.15, 1), loc='upper right')

        # Directions (gradients)
        ax = axs[2, curidx]
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.plot(es._warping_directions[cursymbol], '-o', color=colors[4], label="$C_{attr}$")
        ax.plot(es._warping_inertia[cursymbol], '-o', color=colors[5], label="$C_{iner}$")
        if curidx == 0:
            ax.legend(bbox_to_anchor=(-0.15, 1), loc='upper right')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

def plot_paper1(filename):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    xnew = np.linspace(0, 5, 300)

    fn = Path(__file__).parent / "rsrc" / "example9.txt"
    es = EventSeries.from_file(fn, window=LinearScalingWindow(3), constraints=[])

    fn2 = Path(__file__).parent / "rsrc" / "example9_aligned.txt"
    es2 = EventSeries.from_file(fn2, window=LinearScalingWindow(3), constraints=[])

    symbol = [0, 1]
    es.compute_windowed_counts()
    es.compute_rescaled_counts()
    es.compute_warping_directions()

    es2.compute_windowed_counts()
    es2.compute_rescaled_counts()
    es2.compute_warping_directions()

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row',
                            figsize=(10, 4))
    cnts = es.get_counts(ignore_merged=True)
    cnts2 = es2.get_counts(ignore_merged=True)
    colors = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    for curidx, cursymbol in enumerate(symbol):
        curcnts = cnts[cursymbol]
        curcnts2 = cnts2[cursymbol]

        # initial
        ax = axs[0, curidx]
        if curidx ==0:
            ax.set_ylabel('Before alignment')
        ax.set_title(f"Symbol {es.int2symbol.get(cursymbol, cursymbol)}")
        ax.bar(list(range(len(curcnts))), curcnts/sum(curcnts), color=colors[0], label="Normalized counts")
        smooth = make_interp_spline(range(6), np.convolve(curcnts / sum(curcnts), [0.5, 1, 0.5], 'same') / 2)(xnew)
        ax.plot(xnew, smooth, '-', color=colors[3], label="Density $d_{i,S}$")
        ax.set_ylim(0,1)

        # Aligned
        ax = axs[1, curidx]
        if curidx ==0:
            ax.set_ylabel('After alignment')
        ax.set_xlabel('Timestamp')
        ax.bar(list(range(len(curcnts2))), curcnts2/sum(curcnts2), color=colors[0], label="Normalized counts")
        smooth = make_interp_spline(range(6), np.convolve(curcnts2 / sum(curcnts2), [0.5, 1, 0.5], 'same') / 2)(xnew)
        ax.plot(xnew, smooth, '-', color=colors[3], label="Density $d_{i,S}$")
        ax.set_ylim(0, 1)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    # define data and parameters
    symbol_ordenings = [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]

    items_folder = Path(r"C:\Users\dries\python projects\itemsets_final")  # training and files test set
    items_smooth_folder = Path(r"C:\Users\dries\python projects\smoothed_itemsets_final")
    models_folder = Path(r"C:\Users\dries\python projects\smoothed_itemsets_final")
    names = [i for i in os.listdir(items_folder) if (i.endswith('itemsets.txt')) and not (i.endswith('parsed_itemsets.txt'))]
    names_train = names[::2]
    names_test = names[1::2]

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

    # # correct itemsets for unexpected short drops
    # for k, name in enumerate(names):
    #     fn = items_folder / name
    #     print(f'correct short drops {k}/{len(names)}')
    #     with fn.open("r") as fp:
    #         data = fp.read()
    #     data = ast.literal_eval(data)
    #     for lows, highs in [[{5, 6, 7}, {11, 12, 13, 14}], [{15, 16, 17}, {21, 22, 23, 24}]]:
    #         high_peak = [len(i.intersection(highs)) > 0 for i in data]
    #         low_peak = [len(i.intersection(lows)) > 0 for i in data]
    #         short_drop = (np.convolve(np.array(high_peak)-0.5, [1,-1, 1], 'same') == 1.5) & low_peak
    #         short_drop_ind = np.where(short_drop)[0]
    #         for i in short_drop_ind:
    #             data[i] = data[i].difference(lows).union(data[i-1].intersection(highs))
    #     with fn.open("w") as fp:
    #         fp.write(repr(data))

    #############################################

    # # save smoothed itemsets
    # for k, name in enumerate(names):
    #     print(f'Smoothing {k}/{len(names)}')
    #     setlistfile2setlistsfile(items_folder / name, items_smooth_folder / name, start={1,3}, stop={2,4}, margin=5, symbol_ordenings=symbol_ordenings)

    ##############################################

    # # train event alignment
    train_event_series_alignment([items_smooth_folder / i for i in names_train], param_dict, n_iter, models_folder, file_name_apnea, file_name_hypopnea)

    ##############################################

    with open(models_folder / (file_name_apnea), 'rb') as file:
        es_apnea = pickle.load(file)
    with open(models_folder / (file_name_hypopnea), 'rb') as file:
        es_hypopnea = pickle.load(file)

    # warp test data with model
    # make_test_set([items_folder / i for i in names_test], items_smooth_folder, symbol_ordenings=symbol_ordenings)
    # files_test_data = [items_smooth_folder / 'test_data.txt']
    # warp_test_data_with_model(files_test_data, param_dict, n_iter, es_apnea, es_hypopnea, models_folder, file_name_apnea_model, file_name_hypopnea_model)

    ##############################################

    # # train warping on sequences without apnea or hypopnea
    # files_test_data = [items_smooth_folder / 'test_data.txt']
    # with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
    #     labels_test = np.array(eval(file.read()))
    # train_event_series_alignment_no_apnea(files_test_data, labels_test, param_dict, n_iter, models_folder)

    #############################################

    #Make plots paper

    with open(models_folder / (file_name_apnea), 'rb') as file:
        es_apnea = pickle.load(file)
    with open(models_folder / (file_name_hypopnea), 'rb') as file:
        es_hypopnea = pickle.load(file)
    with open(models_folder / 'es_none', 'rb') as file:
        es_none = pickle.load(file)
    with open(models_folder / file_name_apnea_model, 'rb') as file:
        es_apnea_model = pickle.load(file)
    with open(models_folder / file_name_hypopnea_model, 'rb') as file:
        es_hypopnea_model = pickle.load(file)
    with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
        labels_test = np.array(eval(file.read()))

    # plot_training_examples(es_apnea, es_hypopnea, 'series_examples', 12, 31)
    # plot_warping(es_apnea_model, labels_test, 1, name='apnea_warped_by_apnea')
    # plot_warping(es_hypopnea_model, labels_test, 2, name='hypopnea_warped_by_hypopnea')
    # plot_warping(es_apnea_model, labels_test, 0, name='normal_warped_by_apnea')
    # plot_density([es_apnea, es_hypopnea], 'dist_apnea')
    # plot_density([es_apnea, es_hypopnea,es_none], 'dist_all')
    # density = es_apnea.get_counts(ignore_merged=True) / len(es_apnea.series)
    # plot_single_series(es_apnea_model, 'single_warping', 53, density)
    # density2 = es_hypopnea.get_counts(ignore_merged=True) / len(es_apnea.series)
    # plot_single_series(es_hypopnea_model, 'single_warping2', 53, density2)
    # plot_paper1('plot_paper_density')
    # plot_directions('warping_method')
    ##############################################

    # evaluation on test data set
    with open(models_folder / file_name_apnea_model, 'rb') as file:
        es_apnea_model = pickle.load(file)
    with open(models_folder / file_name_hypopnea_model, 'rb') as file:
        es_hypopnea_model = pickle.load(file)
    # diff_series = series_to_diff_series(es_apnea_model.warped_series, symbol_ordenings)
    # es_apnea_model.diff_series = diff_series

    with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
        labels_test = np.array(eval(file.read()))
    print("----------apnea model-------------")
    do_evaluations(labels_test, es_apnea_model, 'apnea', 1, items_folder, items_smooth_folder)
    plt.show()

    ##############################################

    # # evaluation on individual file
    # ## train clf on test set data
    # with open(models_folder / file_name_apnea_model, 'rb') as file:
    #     model = pickle.load(file)
    # x = np.sign(model.warped_series)
    # x = x.reshape((len(x), -1))
    # with (items_smooth_folder / 'test_data_labels.txt').open('r') as file:
    #     y = np.array(eval(file.read()))
    # clf = RFC(min_samples_leaf=30, max_features=100, n_estimators=100)
    # clf.fit(x, y)
    #
    # # # evaluate on file
    # # folder_test_files = Path(r"C:\Users\dries\python projects\itemsets_multilabeled")
    # # files = [folder_test_files / i for i in os.listdir(folder_test_files) if (i.endswith('itemsets.txt'))]
    # # for i in files:
    # #     brt_file = [j for j in uza_multilabeled_single_files if str(i.parts[-1]).replace('_itemsets.txt','') in j][0]
    # #     pred = predict_apnea_per_file(i, clf, model, uza_multilabeled_single_files[brt_file])
    #
    # path = Path(r"C:\Users\dries\python projects\itemsets_new\DZ_00001_0000414_itemsets.txt")
    # pred = predict_apnea_per_file(path, clf, model)

    print(1)
