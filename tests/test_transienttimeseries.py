from pathlib import Path
import os
import math
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from dtaidistance.preprocessing import differencing
from dtaidistance.dtw import distance_fast
from dtaidistance.subsequence.dtw import subsequence_alignment

from eventwarping.timeseries import featurization
from eventwarping.eventseries import EventSeries


directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))


def test_trace_medoids():
    rsrc = Path(__file__).parent / "rsrc" / "trace"
    fn = rsrc / "Trace_TRAIN.txt"
    data = np.loadtxt(fn)
    data = data[np.argsort(data[:, 0])]
    labels = data[:, 0]
    series = data[:, 1:]

    # Filter data (to speed up experiment)
    nb_occurrences = 3  # occurrence of each class
    data2 = np.zeros((4 * nb_occurrences, data.shape[1]))
    cnts = [0] * (4 + 1)
    for r in range(data.shape[0]):
        label = int(data[r, 0])
        if cnts[label] < nb_occurrences:
            data2[cnts[label] + (label - 1) * nb_occurrences, :] = data[r, :]
            cnts[label] += 1
    data = data2
    print(f"Data: {data.shape}")
    data = data[np.argsort(data[:, 0])]
    labels = data[:, 0]
    series = data[:, 1:]

    # Load motifs
    with (directory / 'motifs.py').open('r') as fp:
        data = fp.read()
    array = np.array
    data = ast.literal_eval(data)
    medoids = data["medoidd"]

    seriesd = differencing(series, smooth=0.1)

    # TODO: this ignores if the same pattern repeats
    patterns = np.zeros((seriesd.shape[0], seriesd.shape[1], len(medoids)))
    for sidx in range(seriesd.shape[0]):
        for midx in range(len(medoids)):
            sa = subsequence_alignment(medoids[midx], seriesd[sidx, :])
            for match in sa.kbest_matches_fast(k=None):
                patterns[sidx, match.segment[0]:match.segment[1], midx] = match.value

    best_patterns = np.argmax(patterns, axis=2)
    print('done')



def test_trace_tsfresh():
    rsrc = Path(__file__).parent / "rsrc" / "trace"
    fn = rsrc / "Trace_TRAIN.txt"
    data = np.loadtxt(fn)
    data = data[np.argsort(data[:, 0])]
    labels = data[:, 0]
    series = data[:, 1:]

    classes = np.unique(labels)
    fig, axs = plt.subplots(nrows=len(classes), ncols=1, sharex=True, sharey=True)
    for classidx, classval in enumerate(classes):
        for r in range(series.shape[0]):
            if labels[r] == classval:
                axs[classidx].plot(series[r, :], alpha=0.5)
    fig.savefig(directory / "traces.png")
    plt.close(fig)

    # Prepare for tsfresh
    nb_series = data.shape[0]
    series_length = data.shape[1] - 1
    slice_length = 50
    nb_events = math.floor(series_length / slice_length)
    slice_features = []
    for slice_idx in range(nb_events):
        slice_start = slice_idx * slice_length
        slice_stop = (slice_idx + 1) * slice_length
        cur_series = series[:, slice_start:slice_stop]
        cur_series = np.reshape(cur_series, np.prod(cur_series.shape))

        ids = np.array(range(nb_series))
        ids = np.tile(np.atleast_2d(ids).T, slice_length)
        ids = np.reshape(ids, np.prod(ids.shape))

        time = np.array(range(ids.shape[0]))
        df = pd.DataFrame({'time': time, 'id': ids, 'y': cur_series})
        extracted_features = extract_features(df, column_id="id", column_sort="time")
        slice_features.append(extracted_features)


    # extracted_features = extract_features(df, column_id="id", column_sort="time")
    # df_rolled = roll_time_series(df, column_id="id", column_sort="time",
    #                              min_timeshift=30, max_timeshift=30)
    # df_features = extract_features(df_rolled, column_id="id", column_sort="time")


    es = EventSeries(window=3, intonly=True)
    nb_symbols = len(slice_features[0].columns)
    es.series = np.zeros((nb_series, nb_events, nb_symbols), dtype=int)
    for eid in range(nb_events):
        f = slice_features[eid]


    print(df.head())

