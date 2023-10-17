import numpy as np
from scipy import signal


class EventSeries:
    def __init__(self):
        self.series = None
        self.alphabet = None
        self.nb_series = 0
        self.nb_events = 0  # length of series
        self.nb_symbols = 0
        self.window = 3
        self.count_thr = 5  # expected minimal number of events
        self._windowed_counts = None
        self._warping_directions = None

    def reset(self):
        self._windowed_counts = None
        self._warping_directions = None

    @staticmethod
    def from_file(fn):
        es = EventSeries()
        es.alphabet = dict()
        allseries = list()
        es.nb_series = 0
        es.nb_symbols = 0
        with fn.open("r") as fp:
            for line in fp.readlines():
                series = []
                es.nb_series += 1
                for events in line.split("|"):
                    events = events.strip()
                    if events != "":
                        events = [e.strip() for e in events.strip().split(" ")]
                        for event in events:
                            if event not in es.alphabet:
                                es.alphabet[event] = es.nb_symbols
                                es.nb_symbols += 1
                    series.append(events)
                if len(series) > es.nb_events:
                    es.nb_events = len(series)
                allseries.append(series)
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=bool)
        for sei, series in enumerate(allseries):
            for evi, events in enumerate(series):
                for symbol in events:
                    syi = es.alphabet[symbol]
                    es.series[sei, evi, syi] = True
        return es

    @staticmethod
    def _compute_windowed_counts(series, nb_series, window):
        """
        Count over window and series
        """
        w = np.lib.stride_tricks.sliding_window_view(series, (nb_series, window), (0, 1))
        w = w.sum(axis=(-2, -1))
        w = np.reshape(w, w.shape[1])
        return w

    @property
    def windowed_counts(self):
        if self._windowed_counts is not None:
            return self._windowed_counts
        self._windowed_counts = self._compute_windowed_counts(self.series, self.nb_series, self.window)
        return self._windowed_counts

    @staticmethod
    def _compute_warping_directions(counts, kernel_width, count_thr):
        """
        Directions in which each series time point should change.

        We use the windowed counts as the density function and use the derivative
        as the direction.

        :return:
        """
        kernel = signal.windows.hann(kernel_width)  # make sure to be uneven to be centered
        # kernel = np.ones(kernel_width) / kernel_width  # make sure to be uneven to be centered
        countsf = signal.convolve(counts, kernel, mode='same') / sum(kernel)
        gradients = np.gradient(countsf, edge_order=1)
        print(f"Gradients: {gradients}")

        # divide by 2 for the gradient function (edge_order=1)
        # divide by kernel_width for convolve function
        part = count_thr / (2 * kernel_width)  # threshold
        directions = gradients / part

        return directions

    @property
    def warping_directions(self):
        if self._warping_directions is not None:
            return self._warping_directions
        # Smooth window to triple its size, a window on each side of the window
        # (but make sure it is uneven to be centered)
        kernel_width = (self.window // 2) * 2 + 1
        self._warping_directions = self._compute_warping_directions(self.windowed_counts,
                                                                    kernel_width, self.count_thr)
        return self._warping_directions

    def print_series(self):
        print(self.series)
        print(self.alphabet)
