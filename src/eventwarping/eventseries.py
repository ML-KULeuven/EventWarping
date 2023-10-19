import numpy as np
from scipy import signal


class EventSeries:
    def __init__(self, window):
        self.series = None  # shape = (nb_series, nb_events, nb_symbols)
        self.symbol2int = None
        self.int2symbol = None
        self.nb_series = 0
        self.nb_events = 0  # length of series
        self.nb_symbols = 0
        self.window = window
        self.count_thr = 5  # expected minimal number of events
        self._windowed_counts = None
        self._warping_directions = None
        self._warped_series = None

    def reset(self):
        self._windowed_counts = None
        self._warping_directions = None
        self._warped_series = None

    def warp(self, iterations=1, restart=False):
        if restart:
            self._warped_series = self.series
        for it in range(iterations):
            self.compute_windowed_counts()
            self.compute_warping_directions()
        return self.warped_series

    @staticmethod
    def from_file(fn, window):
        es = EventSeries(window)
        es.symbol2int = dict()
        es.int2symbol = dict()
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
                            if event not in es.symbol2int:
                                es.symbol2int[event] = es.nb_symbols
                                es.int2symbol[es.nb_symbols] = event
                                es.nb_symbols += 1
                    series.append(events)
                if len(series) > es.nb_events:
                    es.nb_events = len(series)
                allseries.append(series)
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=bool)
        for sei, series in enumerate(allseries):
            for evi, events in enumerate(series):
                for symbol in events:
                    syi = es.symbol2int[symbol]
                    es.series[sei, evi, syi] = True
        return es

    def compute_windowed_counts(self):
        """
        Count over window and series

        :return: Counts (also stored in self.windowed_counts)
        """
        if self._warped_series is None:
            self._warped_series = self.series
        w = np.lib.stride_tricks.sliding_window_view(self._warped_series, (self.nb_series, self.window), (0, 1))
        w = w.sum(axis=(-2, -1))
        # w = np.reshape(w, w.shape[1])
        w = w[0].T
        self._windowed_counts = w
        self._warping_directions = None
        self._warped_series = None
        return self._windowed_counts

    @property
    def windowed_counts(self):
        if self._windowed_counts is not None:
            return self._windowed_counts
        return self.compute_windowed_counts()

    def compute_warping_directions(self):
        """
        Directions in which each series time point should change.

        We use the windowed counts as the density function and use the derivative
        as the direction.

        :return: warping directions (also stored in self.warping_directions)
        """
        # Setup up kernel
        # Smooth window to triple its size, a window on each side of the window
        # (but make sure it is uneven to be centered)
        kernel_width = (self.window // 2) * 2 + 1
        kernel = signal.windows.hann(kernel_width)  # make sure to be uneven to be centered
        # kernel = np.vstack([kernel] * 2)  # kernel per symbol
        # kernel = np.ones(kernel_width) / kernel_width  # make sure to be uneven to be centered

        # Convolve kernel
        countsf = np.zeros(self.windowed_counts.shape)
        for si in range(countsf.shape[0]):
            countsf[si, :] = signal.convolve(self.windowed_counts[si, :], kernel, mode='same') / sum(kernel)
        gradients = np.gradient(countsf, axis=1, edge_order=1)
        print(f"Gradients: {gradients}")

        # Transform gradients into directions (and magnitude)
        # divide by 2 for the gradient function (edge_order=1)
        # divide by kernel_width for convolve function
        part = self.count_thr / (2 * kernel_width)  # threshold
        self._warping_directions = gradients / part

        self._warped_series = None
        return self._warping_directions

    @property
    def warping_directions(self):
        if self._warping_directions is not None:
            return self._warping_directions
        return self.compute_warping_directions()

    def compute_warped_series(self):
        dir = self.warping_directions
        ws = np.zeros((self.nb_series, self.nb_events, self.nb_symbols), dtype=bool)

        for si in range(self.nb_series):

            # Compute aggregated direction
            # TODO

            # Dynamic programming with window the maximal warping direction
            # Window is the deviation from the diagonal (zero is the diagonal)
            # Loss function is sum of changes not followed
            window = max(np.max(dir), -np.min(dir))
            print(f"{window=:.2f}")

            # Do realignment
            # TODO: Should original items sets be remembered or can they be merged

        self._warped_series = ws
        return self._warped_series

    @property
    def warped_series(self):
        if self._warped_series is not None:
            return self._warped_series
        return self.compute_warped_series()

    def print_series(self):
        print(self.series)
        print(self.symbol2int)

    def format_series(self):
        return self._format_series(self.series)

    def format_warped_series(self):
        return self._format_series(self.warped_series)

    def _format_series(self, series):
        # (nb_series, nb_events, nb_symbols)
        s = ''
        for sei in range(self.nb_series):
            for evi in range(self.nb_events):
                s += ' '
                for syi in range(self.nb_symbols):
                    if series[sei, evi, syi]:
                        s += f"{self.int2symbol[syi]} "
                    else:
                        s += "  "
                s += "|"
            s += "\n"
        return s
