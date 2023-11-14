import collections
from pathlib import Path
from typing import Optional
import math

import numpy as np
import numpy.typing as npt
from scipy import signal


V_WC = 0  # Index for version of windowed counts
V_RC = 1  # Index for rescaled counts
V_WD = 2  # Index for version of warping directions
V_WS = 3  # Index for version of warped series


class EventSeries:
    def __init__(self, window=3, intonly=False, constraints=None):
        """Warp series of symbolic events.

        :param window: Window over which can be warped (should be an odd integer).
            Full size of window. For example window=3 means, that symbols one
            slot to the left and one slot to the right are considered.
        :param intonly: No dictionary for symbols used, the symbols are integer indices,
            symbols are integers starting from 0 and range to the largest integer used.
            This is not recommended for sparse sets of integers.
        :param constraints: Child class inheriting from ConstraintsBaseClass
        """
        if window % 2 == 0:
            raise ValueError(f"Argument window should be an uneven number.")

        self.series: Optional[npt.NDArray[np.int]] = None  # shape = (nb_series, nb_events, nb_symbols)
        self.intonly = intonly
        self.symbol2int = dict()
        self.int2symbol = dict()
        self.nb_series = 0
        self.nb_events = 0  # length of series (of sets)
        self.nb_symbols = 0  # maximal size of set that represents an event
        self.window = window
        self.count_thr = 5  # expected minimal number of events
        self.rescale_power = 2  # Raise counts to this power and rescale
        self.rescale_weights = dict()  # Symbol idx to weight factor
        self.rescale_active = True
        self._windowed_counts = None
        self._rescaled_counts = None
        self._warping_directions = None
        self._warping_inertia = None
        self._zero_crossings = None
        self._warped_series: Optional[npt.NDArray[np.int]] = None
        self._versions = [0, 0, 0, 0]  # V_WC, V_RC, V_WD, V_WS
        self.constraints = constraints
        if self.constraints is not None:
            self.constraints.es = self

        # Use the new warping, with constraints built in
        # Otherwise we cannot (?) guarantee that the following does not happen:
        # | A |   | A |
        # to
        # |   | A |   |
        self._use_warping_v2 = False
        self.allow_merge = np.inf  # Limit number of merges to this value (per symbol), should be >0

    def reset(self):
        self._windowed_counts = None
        self._rescaled_counts = None
        self._warping_directions = None
        self._warped_series = None
        self._versions = [0, 0, 0, 0]

    def warp(self, iterations=1, restart=False, plot=None):
        if plot is not None:
            import matplotlib.pyplot as plt
        if restart:
            self.reset()
        for it in range(iterations):
            self.compute_windowed_counts()
            self.compute_rescaled_counts()
            self.compute_warping_directions()
            if plot is not None:
                fig, axs = self.plot_directions(symbol=plot.get("symbol", None),
                                                seriesidx=plot.get("seriesidx", None))
                fig.savefig(plot['filename'].format(it=it))
                plt.close(fig)
            self.compute_warped_series()
        return self.warped_series

    def warp_yield(self, iterations=1, restart=False):
        if restart:
            self.reset()
        for it in range(iterations):
            self.compute_windowed_counts()
            self.compute_rescaled_counts()
            self.compute_warping_directions()
            self.compute_warped_series()
            yield self.warped_series

    @classmethod
    def from_setlistfile(cls, fn, window, intonly=False, constraints=None, max_series_length=None):
        """Read a setlist file and create an eventwarping object.
        A file where each line is a list of sets of symbols. Each set represents a time point.

        :param fn: Filename or Path object
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param constraints: See EventWarping
        :return: EventWarping object
        """
        import ast
        data = list()
        if type(fn) is str:
            fn = Path(fn)
        with fn.open("r") as fp:
            for line in fp.readlines():
                data.append(ast.literal_eval(line))
        return cls.from_setlist(data, window, intonly, constraints, max_series_length)

    @classmethod
    def from_setlistfiles(cls, fns, window, intonly=False, selected=None,
                          constraints=None, max_series_length=None):
        """Read a list of setlist file and create an eventwarping object.
        A file where each line is a list of sets of symbols. Each set represents a time point.

        :param fn: Filename or Path object
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param selected: Boolean list. Only use the i-th series if selected[i] is True
        :param constraints: See EventWarping
        :return: EventWarping object
        """
        import ast
        data = list()
        i = 0
        for fn in fns:
            if type(fn) is str:
                fn = Path(fn)
            with fn.open("r") as fp:
                for line in fp.readlines():
                    if (selected is None) or selected[i]:
                        data_line = ast.literal_eval(line)
                        data.append(data_line)
                    i += 1
        return cls.from_setlist(data, window, intonly, constraints, max_series_length)

    @classmethod
    def from_setlist(cls, sl, window, intonly=False, constraints=None, max_series_length=None):
        """Convert a setlist to an eventwarping object.

        :param sl: List of sets of symbols
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param constraints: See EventWarping
        :param max_series_length: lenght of each series (i.e. #itemsets) is truncated to this size
        :return: EventWarping object
        """
        es = EventSeries(window=window, intonly=intonly, constraints=constraints)
        es.nb_symbols = 0
        es.nb_events = 0
        es.nb_series = len(sl)
        for series in sl:
            if len(series) > es.nb_events:
                es.nb_events = len(series)
            for event in series:
                for symbol in event:
                    if intonly:
                        if type(symbol) is not int:
                            raise ValueError(f"Symbol is not an int: {symbol}")
                        if symbol >= es.nb_symbols:
                            es.nb_symbols = symbol + 1
                    else:
                        if symbol not in es.symbol2int:
                            es.symbol2int[symbol] = es.nb_symbols
                            es.int2symbol[es.nb_symbols] = symbol
                            es.nb_symbols += 1
        if max_series_length:
            es.nb_events = min(es.nb_events, max_series_length)
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=int)
        for sei, series in enumerate(sl):
            for evi, events in enumerate(series[:es.nb_events]):
                for symbol in events:
                    if not intonly:
                        symbol = es.symbol2int[symbol]
                    es.series[sei, evi, symbol] = 1
        return es

    @classmethod
    def from_file(cls, fn, window, intonly=False, constraints=None, max_series_length=None):
        """Convert a simple formatted file to an eventwarping object.

        The format is:
        - A line is a series
        - Events are separated by pipes '|'
        - Symbols are separated by spaces

        For example:

            | A |   | A B |
            |   | A |   B |

        :param fn: Filename or file with list of sets of symbols
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param constraints: See EventWarping
        :return: EventWarping object
        """
        es = EventSeries(window, intonly=intonly, constraints=constraints)
        allseries = list()
        with fn.open("r") as fp:
            for line in fp.readlines():
                series = []
                es.nb_series += 1
                for events in line.split("|"):
                    events = events.strip()
                    if events != "":
                        events = [e.strip() for e in events.strip().split(" ") if e.strip() != ""]
                        for event in events:
                            if intonly:
                                if type(event) is not int:
                                    raise ValueError(f"Symbol is not int: {event}")
                                if event >= es.nb_symbols:
                                    es.nb_symbols = event + 1
                            else:
                                if event not in es.symbol2int:
                                    es.symbol2int[event] = es.nb_symbols
                                    es.int2symbol[es.nb_symbols] = event
                                    es.nb_symbols += 1
                    series.append(events)
                if len(series) > es.nb_events:
                    es.nb_events = len(series)
                allseries.append(series)
        if max_series_length:
            es.nb_events = min(es.nb_events, max_series_length)
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=int)
        for sei, series in enumerate(allseries):
            for evi, events in enumerate(series):
                for symbol in events:
                    syi = symbol if intonly else es.symbol2int[symbol]
                    es.series[sei, evi, syi] = 1
        return es

    def insert_spacers(self, nb_spacers):
        ws = np.zeros((self.nb_series, (1+nb_spacers)*self.nb_events, self.nb_symbols), dtype=int)
        ws[:, 0, :] = self._warped_series[:, 0, :]
        for i in range(1, self.nb_events):
            ws[:, (1+nb_spacers)*i, :] = self._warped_series[:, i, :]
        self.series = ws
        self.nb_events = (1+nb_spacers)*self.nb_events
        self.window = ((self.window // 2) + nb_spacers)*2 + 1
        self.reset()

    def compute_counts(self):
        if self._warped_series is None:
            self._warped_series = self.series
        cnts = self._warped_series.sum(axis=0).T
        return cnts

    def compute_windowed_counts(self):
        """Count over window and series

        :return: Counts (also stored in self.windowed_counts)
        """
        if self._warped_series is None:
            self._warped_series = self.series
        # Only count occurrence of a symbol in a series once, even though
        # we keep track of how many are merged.
        ws = np.sign(self._warped_series)
        # Sliding window to aggregate from neighbours
        w = np.lib.stride_tricks.sliding_window_view(ws, (self.nb_series, self.window), (0, 1))
        sides = int((self.window - 1) / 2)
        wc = np.zeros((self.nb_symbols, self.nb_events))
        wc[:, sides:-sides] = w.sum(axis=(-2, -1))[0].T
        # Pad the beginning and ending with the same values (having half a window can lower the count)
        wc[:, :sides] = wc[:, sides:sides+1]
        wc[:, -sides:] = wc[:, -sides-1:-sides]
        # Add without a window (otherwise the begin and end cannot differentiate)
        # c = self._warped_series.sum(axis=0).T
        # self._windowed_counts = np.add(wc, c) / 2
        self._windowed_counts = wc
        self._versions[V_WC] += 1
        return self._windowed_counts

    @property
    def windowed_counts(self):
        if self._windowed_counts is not None:
            return self._windowed_counts
        return self.compute_windowed_counts()

    def compute_rescaled_counts(self):
        """Rescale the counts such that for each symbol the counts add up to one.
        The rescaling is done by first taking the power of the values to give peaks
        more weight.
        """
        if self._versions[V_RC] == self._versions[V_WC]:
            self.compute_windowed_counts()
        if not self.rescale_active:
            self._rescaled_counts = self._windowed_counts
            return self._rescaled_counts

        # Normalize per symbol
        countsp = np.power(self.windowed_counts, self.rescale_power)
        sums = countsp.sum(axis=1)
        sums[sums == 0.0] = 1.0
        countsp = countsp / sums[:, np.newaxis]

        # Reweigh certain symbols
        for symbol, weight in self.rescale_weights.items():
            countsp[symbol, :] *= weight

        self._rescaled_counts = countsp

        self._versions[V_RC] += 1
        return self._rescaled_counts

    @property
    def rescaled_counts(self):
        if self._rescaled_counts is not None:
            return self._rescaled_counts
        return self.compute_rescaled_counts()

    def compute_warping_directions(self):
        """
        Directions in which each series time point should change.

        We use the windowed counts as the density function and use the derivative
        as the direction.

        :return: warping directions (also stored in self.warping_directions)
        """
        # If windowed_counts has not yet been recomputed, do so
        if self._versions[V_WD] == self._versions[V_RC]:
            self.compute_rescaled_counts()

        # Setup up kernel
        # Smooth window to triple its size, a window on each side of the window
        # (but make sure it is uneven to be centered)
        kernel_width = (self.window // 2) * 2 + 1
        kernel = signal.windows.hann(kernel_width)  # make sure to be uneven to be centered
        # kernel = np.vstack([kernel] * 2)  # kernel per symbol
        # kernel = np.ones(kernel_width) / kernel_width  # make sure to be uneven to be centered

        # Convolve kernel
        counts = self._rescaled_counts
        countsf = np.zeros(counts.shape)
        for si in range(countsf.shape[0]):
            countsf[si, :] = signal.convolve(counts[si, :], kernel, mode='same') / sum(kernel)
        gradients = np.gradient(countsf, axis=1, edge_order=1)

        # Transform gradients into directions (and magnitude)
        # divide by 2 for the gradient function (edge_order=1)
        # divide by kernel_width for convolve function
        if not self.rescale_active:
            part = self.count_thr / (2 * kernel_width)  # threshold
            self._warping_directions = gradients / part
        else:
            self._warping_directions = gradients
        # The warping direction only expresses the direction using the sign. The
        # number is the weight whether this move should be preferred. But every
        # move is just one step (otherwise it overshoots peaks).

        # Only retain large enoug directions (depends on self.count_thr)
        # self._warping_directions[~((self._warping_directions > 1) | (self._warping_directions < -1))] = 0

        # Warping should not be beyond peak in gradients? Makes the
        # symbols swap and will not converge
        # We avoid contractions (this is when the gradients are pos and then neg)
        conti = np.where(np.diff(np.sign(self._warping_directions)) == -2)
        for idx, (r, c) in enumerate(zip(*conti)):
            if abs(self._warping_directions[r, c]) > abs(self._warping_directions[r, c + 1]):
                conti[1][idx] += 1
        self._warping_directions[conti] = 0
        # All zero crossings from pos to neg have inertia, we don't want to move them as they are already a peak
        self._warping_inertia = np.zeros(self._warping_directions.shape)
        conti = np.where(self._warping_directions == 0.0)
        for r, c in zip(*conti):
            if c == 0 or c == self._warping_directions.shape[1] - 1:
                continue
            if self._warping_directions[r, c - 1] > 0 > self._warping_directions[r, c + 1]:
                self._warping_inertia[r, c] = max(self._warping_directions[r, c - 1],
                                                  self._warping_directions[r, c + 1])
        self._versions[V_WD] += 1
        return self._warping_directions

    @property
    def warping_directions(self):
        if self._warping_directions is not None:
            return self._warping_directions
        return self.compute_warping_directions()

    @property
    def warping_inertia(self):
        if self._warping_inertia is not None:
            return self._warping_inertia
        return self.compute_warping_directions()

    def _best_warped_path(self, cc, ps=None):
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

        if self._use_warping_v2:
            if ps is None:
                raise ValueError("Warping v2 requires a ps argument")
            for r in reversed(range(cc.shape[0] - 1)):
                prev_c = ps[r+1, prev_c]
                if prev_c == -1:
                    return None
                path.append((r, r + prev_c - 1))
            path.reverse()
            return path

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
                if cc[r, 0] < cost:
                    prev_c, cost = 0, cc[r, 0]
            else:
                raise Exception(f"{prev_c=}")
            path.append((r, r + prev_c - 1))
        path.reverse()
        return path

    def compute_warped_series(self):
        """Warp events by maximally one step (left, right, or none)."""
        # If warping_directions has not yet been recomputed, do so
        if self._versions[V_WD] == self._versions[V_WS]:
            self.compute_warping_directions()

        ws = np.zeros((self.nb_series, self.nb_events, self.nb_symbols), dtype=int)

        # compute constraints
        if self.constraints is not None:
            constraint_matrix = self.constraints.calculate_constraint_matrix()
        else:
            constraint_matrix = np.zeros((self.nb_series, self.nb_events, 3), dtype=bool)

        # Dynamic programming with window size 3. We only allow a shift of one or zero.
        cc = np.zeros((self.nb_events, 3))  # cumulative cost
        if self._use_warping_v2:
            ps = np.zeros((self.nb_events, 3), dtype=int)  # previous state
            sc = np.zeros((self.nb_events, 3, self.nb_symbols), dtype=int)  # summed counts

        for sei in range(self.nb_series):
            # Aggregated direction
            wss = self.warped_series[sei, :, :]
            wss = np.multiply(wss.T, self._warping_directions)
            wss = wss.sum(axis=0)
            # Aggregated inertia
            wsi = self.warped_series[sei, :, :]
            wsi = np.multiply(wsi.T, self._warping_inertia)
            wsi = wsi.sum(axis=0)

            # Initialize datastructures
            cc[:, :] = 0
            if self._use_warping_v2:
                ps[:, :] = 0
                sc[:, :, :] = 0

            # Initialize first row
            cc[0, 0] = np.inf   # First element cannot move backward
            cc[0, 1] = 0
            cc[0, 2] = -wss[0]
            if self._use_warping_v2:
                ps[0, :] = [0, 1, 2]
                sc[0, 0, :] = 0
                sc[0, 1, :] = self._warped_series[sei, 0, :]
                sc[0, 2, :] = self._warped_series[sei, 0, :]
            else:
                ps = None
                sc = None
            for i in range(1, len(wss)):
                # Backward
                if not self._use_warping_v2:
                    #              Stay one behind
                    #              |           Move on back from diagonal
                    #              |           |             Cost to move backward is the positive gradient
                    cc[i, 0] = min(cc[i-1, 0], cc[i-1, 1]) + wss[i] if not constraint_matrix[sei, i, 0] else np.inf
                else:
                    cc[i, 0] = np.inf
                    ps[i, 1] = -1
                    for prevs in [1, 0]:
                        if prevs == 1:
                            merged_cnts = sc[i-1, prevs] + self._warped_series[sei, i, :]
                        else:
                            merged_cnts = self._warped_series[sei, i, :]
                        if cc[i - 1, prevs] < cc[i, 0] and np.all(merged_cnts <= self.allow_merge):
                            cc[i, 0] = cc[i - 1, prevs]
                            ps[i, 0] = prevs
                            sc[i, 0] = merged_cnts
                    cc[i, 0] += wss[i]
                # Stay
                if not self._use_warping_v2:
                    #              Move back to diagonal from one behind
                    #              |           Stay on diagonal
                    #              |           |           Move to diagonal from one ahead (thus stay)
                    #              |           |           |                Inertia (reward for staying)
                    cc[i, 1] = min(cc[i-1, 0], cc[i-1, 1], cc[i-1, 2]) - wsi[i] if not constraint_matrix[sei, i, 1] else np.inf
                else:
                    cc[i, 1] = np.inf
                    ps[i, 1] = -1
                    for prevs in [1, 0, 2]:
                        if prevs == 2:
                            merged_cnts = sc[i - 1, prevs] + self._warped_series[sei, i, :]
                        else:
                            merged_cnts = self._warped_series[sei, i, :]
                        if cc[i - 1, prevs] < cc[i, 1] and np.all(merged_cnts <= self.allow_merge):
                            cc[i, 1] = cc[i - 1, prevs]
                            ps[i, 1] = prevs
                            sc[i, 1] = merged_cnts
                    cc[i, 1] += -wsi[i]
                # Forward
                if not self._use_warping_v2:
                    #              Skip diagonal and move from one back for previous to one ahead for this one
                    #              |           Move one forward from diagonal
                    #              |           |           Stay one forward
                    #              |           |           |             Cost to move forward is the negative gradient
                    cc[i, 2] = min(cc[i-1, 0], cc[i-1, 1], cc[i-1, 2]) + -wss[i] if not constraint_matrix[sei, i, 2] else np.inf
                else:
                    cc[i, 2] = np.inf
                    ps[i, 2] = -1
                    for prevs in [1, 0, 2]:
                        if cc[i - 1, prevs] < cc[i, 2]:
                            cc[i, 2] = cc[i - 1, prevs]
                            ps[i, 2] = prevs
                            sc[i, 2] = self._warped_series[sei, i, :]
                    cc[i, 2] += -wss[i]
            cc[len(wss) - 1, 2] = np.inf  # Last element cannot move forward
            path = self._best_warped_path(cc, ps)
            if path is None:
                print(f"No path found for series {sei}")
                continue

            # Do realignment
            for i_from, i_to in path:
                ws[sei, i_to, :] = ws[sei, i_to, :] + self._warped_series[sei, i_from, :]

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
        if self.intonly:
            sl = math.floor(math.log(self.nb_symbols - 1, 10)) + 1 + 1

            def fmt_symbol(syi):
                return f"{syi:>{sl}}"
        else:
            sl = max(len(str(k)) for k in self.symbol2int.keys()) + 1

            def fmt_symbol(syi):
                return f"{self.int2symbol[syi]:>{sl}}"
        empty = " " * sl
        s = ''
        for sei in range(self.nb_series):
            for evi in range(self.nb_events):
                for syi in range(self.nb_symbols):
                    if series[sei, evi, syi]:
                        s += fmt_symbol(syi)
                    else:
                        s += empty
                s += " |"
            s += "\n"
        return s

    def plot_directions(self, symbol=0, seriesidx=None, filename=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        if type(symbol) is set:
            pass
        elif type(symbol) is int:
            symbol = {symbol}
        elif isinstance(symbol, collections.Iterable):
            symbol = set(symbol)
        nrows = 2
        if self.rescale_active:
            nrows += 1
        if seriesidx is not None:
            nrows += 1
            wss = np.multiply(self.warped_series[seriesidx, :, :].T, self.warping_directions).sum(axis=0)
            wsi = np.multiply(self.warped_series[seriesidx, :, :].T, self.warping_inertia).sum(axis=0)
        else:
            wss = None
            wsi = None
        fig, axs = plt.subplots(nrows=nrows, ncols=len(symbol), sharex=True, sharey='row', figsize=(5*len(symbol), 4))
        cnts = self.compute_counts()
        # colors = mpl.cm.get_cmap().colors
        colors = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
        # amp = np.max(np.abs(self.warping_directions[list(symbol)]))
        for curidx, cursymbol in enumerate(symbol):
            curcnts = cnts[cursymbol]
            # Counts
            axrow = 0
            ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
            ax.set_title(f"Symbol {cursymbol}: {self.int2symbol.get(cursymbol, cursymbol)}")
            ax.bar(list(range(len(curcnts))), curcnts, color=colors[0], label="Counts")
            ax.plot(self.windowed_counts[cursymbol], '-o', color=colors[1], label="Counts (smoothed)")
            if curidx == 0:
                ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
            axrow += 1
            # Scaled counts
            if self.rescale_active:
                ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
                ax.plot(self.rescaled_counts[cursymbol], '-o', color=colors[2], label="Rescaled counts")
                if curidx == 0:
                    ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
                axrow += 1
            # Directions (gradients)
            ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
            ax.axhline(y=0, color='r', linestyle=':', alpha=0.3)
            # ax.axhline(y=1, color='b', linestyle=':', alpha=0.3)
            # ax.axhline(y=-1, color='b', linestyle=':', alpha=0.3)
            ax.plot(self._warping_directions[cursymbol], '-o', color=colors[3], label="Directions")
            ax.plot(self._warping_inertia[cursymbol], '-o', color=colors[4], label="Inertia")
            # ax.set_ylim(-amp, amp)
            if curidx == 0:
                ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
            axrow += 1
            # Counts in given series
            if seriesidx is not None:
                ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
                cursymbol_cnts = self.warped_series[seriesidx, :, cursymbol]
                ax.axhline(y=0, color='r', linestyle=':', alpha=0.3)
                ax.bar(range(len(cursymbol_cnts)), cursymbol_cnts, label=f"Counts for series {seriesidx}")
                ax.plot(wss, '-+', color=colors[3], label=f"Agg directions for series {seriesidx}")
                ax.plot(wsi, '-+', color=colors[4], label=f"Agg Inertia for series {seriesidx}")
                if curidx == 0:
                    ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
                axrow += 1
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig, axs
