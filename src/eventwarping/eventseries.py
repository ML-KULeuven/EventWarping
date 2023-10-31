import numpy as np
from scipy import signal


V_WC = 0  # Index for version of windowed counts
V_WD = 1  # Index for version of warping directions
V_WS = 2  # Index for version of warped series


class EventSeries:
    def __init__(self, window=3, intonly=False):
        """Warp series of symbolic events.

        :param window: Window over which can be warped (should be an odd integer).
            Full size of window. For example window=3 means, that symbols one
            slot to the left and one slot to the right are considered.
        :param intonly: No dictionary for symbols required,
            symbols are integers starting from 0
        """
        if window % 2 == 0:
            raise ValueError(f"Argument window should be an uneven number.")

        self.series = None  # shape = (nb_series, nb_events, nb_symbols)
        self.intonly = intonly
        self.symbol2int = dict()
        self.int2symbol = dict()
        self.nb_series = 0
        self.nb_events = 0  # length of series
        self.nb_symbols = 0
        self.window = window
        self.count_thr = 5  # expected minimal number of events
        self._windowed_counts = None
        self._warping_directions = None
        self._warped_series = None
        self._versions = [0, 0, 0]  # V_WC, V_WD, V_WS

    def reset(self):
        self._windowed_counts = None
        self._warping_directions = None
        self._warped_series = None
        self._versions = [0, 0, 0]

    def warp(self, iterations=1, restart=False):
        # TODO: nb iterations should be less than window?
        if restart:
            self.reset()
        for it in range(iterations):
            self.compute_windowed_counts()
            self.compute_warping_directions()
            self.compute_warped_series()
        return self.warped_series

    def warp_yield(self, iterations=1, restart=False):
        if restart:
            self.reset()
        for it in range(iterations):
            self.compute_windowed_counts()
            self.compute_warping_directions()
            self.compute_warped_series()
            yield self.warped_series

    @classmethod
    def from_setlistfile(cls, fn, window, intonly=False):
        import ast
        data = list()
        with fn.open("r") as fp:
            for line in fp.readlines():
                data.append(ast.literal_eval(line))
        return cls.from_setlist(data, window, intonly)

    @classmethod
    def from_setlist(cls, sl, window, intonly=False):
        es = EventSeries(window=window, intonly=intonly)
        if not intonly:
            raise AttributeError("Not yet supported")
        max_int = 0
        max_events = 0
        for series in sl:
            if len(series) > max_events:
                max_events = len(series)
            for event in series:
                for symbol in event:
                    if type(symbol) is not int:
                        raise AttributeError(f"Value is not an int: {symbol}")
                    if symbol > max_int:
                        max_int = symbol
        es.nb_series = len(sl)
        es.nb_events = max_events
        es.nb_symbols = max_int + 1
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=bool)
        for sei, series in enumerate(sl):
            for evi, events in enumerate(series):
                for symbol in events:
                    es.series[sei, evi, symbol] = True
        return es

    @classmethod
    def from_file(cls, fn, window):
        es = EventSeries(window)
        allseries = list()
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

    def compute_counts(self):
        if self._warped_series is None:
            self._warped_series = self.series
        cnts = self._warped_series.sum(axis=0).T
        return cnts

    def compute_windowed_counts(self):
        """
        Count over window and series

        :return: Counts (also stored in self.windowed_counts)
        """
        if self._warped_series is None:
            self._warped_series = self.series
        w = np.lib.stride_tricks.sliding_window_view(self._warped_series, (self.nb_series, self.window), (0, 1))
        sides = int((self.window - 1) / 2)
        wc = np.zeros((self.nb_symbols, self.nb_events))
        wc[:, sides:-sides] = w.sum(axis=(-2, -1))[0].T
        # Pad the beginning and ending with the same values (having half a window can lower the count)
        wc[:, :sides] = wc[:, sides:sides+1]
        wc[:, -sides:] = wc[:, -sides-1:-sides]
        # Add without a window (otherwise the begin and end cannot differentiate)
        c = self._warped_series.sum(axis=(0)).T
        self._windowed_counts = np.add(wc, c) / 2
        self._versions[V_WC] += 1
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

        # Transform gradients into directions (and magnitude)
        # divide by 2 for the gradient function (edge_order=1)
        # divide by kernel_width for convolve function
        part = self.count_thr / (2 * kernel_width)  # threshold
        self._warping_directions = gradients / part
        # The warping direction only expresses the direction using the sign. The
        # number is the weight whether this move should be preferred. But every
        # move is just one step (otherwise it overshoots peaks).

        # Only retain large enoug directions (depends on self.count_thr)
        # self._warping_directions[~((self._warping_directions > 1) | (self._warping_directions < -1))] = 0

        # Warping should not be beyond peak in gradients? Makes the
        # symbols swap and will not converge
        # We avoid contractions (this is when the gradients are pos and then neg)
        conti = np.where(np.diff((self._warping_directions >= 0).astype(int)) == -1)
        c = conti[1]
        c += 1
        self._warping_directions[conti] = 0

        self._versions[V_WD] += 1
        return self._warping_directions

    @property
    def warping_directions(self):
        if self._warping_directions is not None:
            return self._warping_directions
        return self.compute_warping_directions()

    def _best_warped_path(self, cc):
        """

        :param cc: Cumulative Cost matrix
        :return: Path [(from, to)]
        """
        r = cc.shape[0] - 1
        # prefer no change
        prev_c, cost = 1, cc[r, 1]
        if cc[r, 0] < cost:
            prev_c, cost = 0, cc[r, 0]
        if cc[r, 2] < cost:
            prev_c, cost = 2, cc[r, 2]
        path = [(r, r + prev_c - 1)]

        for r in reversed(range(cc.shape[0] - 1)):
            if prev_c == 0:
                prev_c, cost = 1, cc[r, 1]
                if cc[r, 0] < cost:
                    prev_c, cost = 0, cc[r, 0]
            elif prev_c == 1:
                prev_c, cost = 1, cc[r, 1]
                if cc[r, 0] < cost:
                    prev_c, cost = 0, cc[r, 0]
                if cc[r, 2] < cost:
                    prev_c, cost = 2, cc[r, 2]
            elif prev_c == 2:
                prev_c, cost = 1, cc[r, 1]
                if cc[r, 2] < cost:
                    prev_c, cost = 2, cc[r, 2]
            else:
                raise Exception(f"{prev_c=}")
            path.append((r, r + prev_c - 1))
        path.reverse()
        return path

    def compute_warped_series(self):
        """Warp events by maximally one step (left, right, or none)."""
        dir = self.warping_directions
        ws = np.zeros((self.nb_series, self.nb_events, self.nb_symbols), dtype=bool)

        for sei in range(self.nb_series):

            # Compute aggregated direction
            wss = self.warped_series[sei, :, :]
            wss = np.multiply(wss.T, self.warping_directions)
            wss = wss.sum(axis=0)

            # Dynamic programming with window size 3. We only allow a shift of one or zero.
            cc = np.zeros((self.nb_events, 3))  # cumulative cost
            cc[0, 0] = np.inf  # first element cannot move backward
            cc[0, 1] = 0
            cc[0, 2] = -wss[0]  # cost to move forward
            for i in range(1, len(wss)):
                # backward
                cc[i, 0] = min(cc[i-1, 0], cc[i-1, 1]) + wss[i]
                # stay
                cc[i, 1] = min(cc[i-1, 0], cc[i-1, 1], cc[i-1, 2])
                # forward
                cc[i, 2] = min(cc[i-1, 1], cc[i-1, 2]) + -wss[i]
            cc[len(wss) - 1, 2] = np.inf
            path = self._best_warped_path(cc)

            # Do realignment
            # TODO: Should original items sets be remembered or can they be merged
            # TODO: Should it be allowed to merge two symbols (changes the counts)
            for i_from, i_to in path:
                ws[sei, i_to, :] = ws[sei, i_to, :] | self.warped_series[sei, i_from, :]

        self._warped_series = ws
        self._versions[V_WS] += 1
        return self._warped_series

    @property
    def warped_series(self):
        if self._warped_series is not None:
            return self._warped_series
        self._warped_series = self.series
        return self._warped_series

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
                        if self.intonly:
                            if syi < 10:
                                s += " "
                            s += str(syi)
                        else:
                            s += f"{self.int2symbol[syi]} "
                    else:
                        s += "  "
                s += "|"
            s += "\n"
        return s

    def plot_directions(self, symbol=0):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 4))
        cnts = self.compute_counts()
        cnts = cnts[symbol]
        ax = axs[0]
        ax.bar(list(range(len(cnts))), cnts, label="Counts")
        ax.legend()
        ax = axs[1]
        ax.axhline(y=0, color='r', linestyle=':', alpha=0.3)
        ax.axhline(y=1, color='b', linestyle=':', alpha=0.3)
        ax.axhline(y=-1, color='b', linestyle=':', alpha=0.3)
        ax.plot(self.windowed_counts[symbol], '-o', label="Counts (smoothed)")
        ax.plot(self.warping_directions[symbol], '-o', label="Directions")
        ax.legend()
        return fig, axs
