from pathlib import Path
from typing import Optional
from collections.abc import Iterable
import math

import numpy as np
import numpy.typing as npt
from scipy import signal, special

from .window import Window


# Index for version of
V_WC = 0  # windowed counts
V_RC = 1  # rescaled counts
V_WD = 2  # warping directions
V_WS = 3  # warped series


class EventSeries:
    def __init__(self, window=3, intonly=False, constraints=None):
        """Warp series of symbolic events.

        :param window: Window over which can be warped (should be an odd integer).
            Full size of window. For example window=3 means, that symbols one
            slot to the left and one slot to the right are considered
        :param intonly: No dictionary for symbols used, the symbols are integer indices,
            symbols are integers starting from 0 and range to the largest integer used.
            This is not recommended for sparse sets of integers
        :param constraints: List of objects inheriting from ConstraintsBaseClass
            See classes in eventwarping.Constraints
        """
        self.series: Optional[npt.NDArray[np.int]] = None  # shape = (nb_series, nb_events, nb_symbols)
        self.intonly = intonly
        self.symbol2int = dict()
        self.int2symbol = dict()
        self.nb_series = 0
        self.nb_events = 0  # length of series (of sets)
        self.nb_symbols = 0  # maximal size of set that represents an event
        self.window = Window.wrap(window)
        self.count_thr = 5  # expected minimal number of events
        self.rescale_power = 1  # Raise counts to this power and rescale
        self.rescale_weights = dict()  # Symbol idx to weight factor
        self.rescale_active = True
        self._windowed_counts = None
        self._factors = None
        self._rescaled_counts = None
        self._smoothed_counts = None
        self.gradients = None
        self.reward_backward = None
        self.reward_forward = None
        self._zero_crossings = None
        self._warped_series: Optional[npt.NDArray[np.int]] = None
        self._warped_series_ec: Optional[npt.NDArray[np.int]] = None  # Event counts
        self._converged = None
        self.model = None
        self._loglikelihoods_p = None
        self._loglikelihoods_n = None
        self._versions = [0, 0, 0, 0]  # V_WC, V_RC, V_WD, V_WS
        self.constraints = constraints
        self.costs = []
        if self.constraints is not None:
            for constraint in self.constraints:
                constraint.es= self

    def reset(self):
        self._windowed_counts = None
        self._rescaled_counts = None
        self.gradients = None
        self.reward_backward = None
        self.reward_forward = None
        self._warped_series = None
        self._warped_series_ec = None
        self._converged = None
        self._versions = [0, 0, 0, 0]

    def series_changed(self):
        """Update derived properties after the series changed."""
        if self.constraints is not None:
            for constraint in self.constraints:
                constraint.es = self

    def warp(self, iterations=10, restart=False, plot=None, window=None):
        for _ in self.warp_yield(iterations=iterations, restart=restart, plot=plot, window=window):
            pass
        return self.warped_series

    def warp_yield(self, iterations=10, restart=False, plot=None, window=None):
        if plot is not None:
            import matplotlib.pyplot as plt
        else:
            plt = None
        if window is not None:
            self.window = window
        if restart:
            self.reset()
        it = 0
        max_iterations = iterations
        iterations_until_converged = 0
        while it <= max_iterations:
            if self._converged is not None or it == max_iterations:
                if self._converged is not None:
                    iterations_until_converged += self._converged
                else:
                    iterations_until_converged += max_iterations
                if self.window.next_window():
                    iterations = max_iterations
                    max_iterations += iterations
                    self.isconverged = False
                else:
                    self.isconverged = iterations_until_converged
                    break
            self.compute_windowed_counts()
            self.compute_rescaled_counts()
            self.compute_rewards()
            if plot is not None:
                fig, axs = self.plot_rewards(symbol=plot.get("symbol", None),
                                             seriesidx=plot.get("seriesidx", None))
                fig.savefig(plot['filename'].format(iteration=it, it=it), bbox_inches='tight')
                plt.close(fig)
            self.compute_warped_series()
            yield self.warped_series
            it += 1

    def warp_with_model(self, model: 'EventSeries' = None, iterations=10):
        """Warp this EventSeries based on the gradients in the given model."""
        if model is None:
            model = self.model
        self._warped_series, _, self._converged = model.align_series_times(self.warped_series, iterations=iterations)
        self._versions[V_WS] += iterations if self._converged is None else self._converged

    @property
    def converged_str(self):
        if self._converged is None:
            return f"Warping did not yet converge after {self._versions[V_WS]} iterations."
        return f"Warping converged after {self._converged+1} iterations."

    @classmethod
    def from_setlistfile(cls, fn, window, intonly=False, constraints=None, max_series_length=None,
                         model: 'EventSeries' = None):
        """Read a setlist file and create an eventwarping object.
        A file where each line is a list of sets of symbols (using Python syntax).
        Each set represents a time point.

        :param fn: Filename or Path object
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param constraints: See EventWarping
        :param max_series_length:
        :param model: Use dictionary from the given EventWarping object
        :return: EventWarping object
        """
        import ast
        data = list()
        if type(fn) is str:
            fn = Path(fn)
        with fn.open("r") as fp:
            for line in fp.readlines():
                data.append(ast.literal_eval(line))
        return cls.from_setlist(data, window, intonly, constraints, max_series_length, model=model)

    @classmethod
    def from_setlistfiles(cls, fns, window, intonly=False, selected=None,
                          constraints=None, max_series_length=None, model: 'EventSeries'=None, n_iter=None):
        """Read a list of setlist file and create an eventwarping object.
        A file where each line is a list of sets of symbols (using Python syntax).
        Each set represents a time point.

        :param fns: List of filenames or Path objects
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param selected: Boolean list. Only use the i-th series if selected[i] is True
        :param constraints: See EventWarping
        :param max_series_length: See from_setlist
        :param model:
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
        return cls.from_setlist(data, window, intonly, constraints, max_series_length, model=model)

    @classmethod
    def from_setlist(cls, sl, window=None, intonly=False, constraints=None, max_series_length=None,
                     model: 'EventSeries' = None):
        """Convert a setlist (a Python list of lists of sets of symbols) to an eventwarping object.

        :param sl: List of lists of sets of symbols
        :param window: See EventWarping
        :param intonly: See EventWarping
        :param constraints: See EventWarping
        :param max_series_length: Length of each series (i.e. #itemsets) is truncated to this size
        :param model: Parse based on given EventSeries object
        :return: EventWarping object
        """
        if window is None:
            if model is None:
                raise ValueError(f"window argument cannot be None if model is None")
            window = model.window
        es = EventSeries(window=window, intonly=intonly, constraints=constraints)
        es.nb_symbols = 0
        es.nb_events = 0
        es.nb_series = len(sl)
        if model is not None:
            es.model = model
            es.nb_symbols = model.nb_symbols
            es.nb_events = model.nb_events
            es.symbol2int = model.symbol2int
            es.int2symbol = model.int2symbol
        for series in sl:
            if len(series) > es.nb_events:
                if es.model is None:
                    es.nb_events = len(series)
                else:
                    raise ValueError(f"setlist is not compatible with the model: "
                                     f"nb_events={len(series)} >= {es.nb_events}")
            for event in series:
                for symbol in event:
                    EventSeries.parse_symbol(symbol, es, intonly)
        if max_series_length and es.model is None:
            es.nb_events = min(es.nb_events, max_series_length)
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=int)
        for sei, series in enumerate(sl):
            for evi, events in enumerate(series[:es.nb_events]):
                for symbol in events:
                    if not intonly:
                        symbol = es.symbol2int[symbol]
                    es.series[sei, evi, symbol] = 1
        es.series_changed()
        return es

    @classmethod
    def from_chararrays(cls, chararrays, window, constraints=None, max_series_length=None,
                        model: 'EventSeries' = None):
        """Create EventSeries from a list of strings. In each string each character is a symbol.

        For example:

            data = ["ABC", "ACC"]

        """
        setlist = [[(symbol,) for symbol in list(seq)] for seq in chararrays]
        return cls.from_setlist(setlist, window, constraints=constraints,
                                max_series_length=max_series_length, model=model)

    @staticmethod
    def parse_symbol(symbol, es, intonly):
        if intonly:
            if type(symbol) is not int:
                raise ValueError(f"Symbol is not int: {symbol}")
            if symbol >= es.nb_symbols:
                if es.model is None:
                    es.nb_symbols = symbol + 1
                else:
                    raise ValueError(
                        f"String is not compatible with the model: "
                        f"nb_events >= {es.nb_events}")
        else:
            if symbol not in es.symbol2int:
                if es.model is not None:
                    raise ValueError(
                        f"String is not compatible with the model: "
                        f"unknown symbol {symbol}")
                es.symbol2int[symbol] = es.nb_symbols
                es.int2symbol[es.nb_symbols] = symbol
                es.nb_symbols += 1

    @classmethod
    def from_filepointer(cls, fp, window=None, intonly=False, constraints=None, max_series_length=None,
                         model: 'EventSeries' = None):
        """See from_file."""
        if window is None:
            if model is None:
                raise ValueError(f"window argument cannot be None if model is None")
            window = model.window
        es = EventSeries(window, intonly=intonly, constraints=constraints)
        if model is not None:
            es.model = model
            es.nb_symbols = model.nb_symbols
            es.nb_events = model.nb_events
            es.int2symbol = model.int2symbol
            es.symbol2int = model.symbol2int
        allseries = list()
        for line in fp.readlines():
            series = []
            es.nb_series += 1
            for events in line.split("|"):
                events = events.strip()
                if events != "":
                    events = [e.strip() for e in events.strip().split(" ") if e.strip() != ""]
                    for symbol in events:
                        EventSeries.parse_symbol(symbol, es, intonly)
                series.append(events)
            if len(series) > es.nb_events:
                if es.model is None:
                    es.nb_events = len(series)
                else:
                    raise ValueError(f"String is not compatible with model: "
                                     f"nb_events={len(series)} >= {es.nb_events}")
            allseries.append(series)
        if max_series_length and es.model is None:
            es.nb_events = min(es.nb_events, max_series_length)
        es.series = np.zeros((es.nb_series, es.nb_events, es.nb_symbols), dtype=int)
        for sei, series in enumerate(allseries):
            for evi, events in enumerate(series):
                for symbol in events:
                    syi = symbol if intonly else es.symbol2int[symbol]
                    es.series[sei, evi, syi] = 1
        es.series_changed()
        return es

    @classmethod
    def from_file(cls, fn, window, intonly=False, constraints=None, max_series_length=None,
                  model: 'EventSeries' = None):
        """Convert a simple formatted file to an EventWarping object.

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
        :param max_series_length: Length of each series (i.e. #itemsets) is truncated to this size
        :param model: Parse based on an existing EventWarping object
            (this uses the already existing symbol dictionary)
        :return: EventWarping object
        """
        with fn.open("r") as fp:
            es = cls.from_filepointer(fp, window, intonly, constraints, max_series_length, model=model)
        return es

    @classmethod
    def from_string(cls, string, window=None, intonly=False, constraints=None, max_series_length=None, model=None):
        """See from_file."""
        import io
        fp = io.StringIO(string)
        es = cls.from_filepointer(fp, window, intonly, constraints, max_series_length, model)
        return es

    def copy(self, filter_symbols=None):
        es = EventSeries(window=self.window, intonly=self.intonly, constraints=self.constraints)
        if filter is None:
            es.series = self.series.copy()
        else:
            ridx, _, _ = np.where(self.series[:, :, filter_symbols])
            ridx = np.unique(ridx)
            es.series = self.series[ridx].copy()
        es.nb_series, es.nb_events, es.nb_symbols = es.series.shape
        es.series_changed()
        return es

    def insert_spacers(self, nb_spacers, update_window=True):
        """Introduce empty events in between events.

        This can be used when using the MaxMergeEventConstraint(1), thus when no merging is allowed.
        By moving the empty slots, warping is still possible following a principle of 'bubble warping'.

        For example:

            | A | B | A | A |
            | A | A | B | A |

        Will be aligned to (when no merging is allowed):

            |   | A | B | A | A |   |   |   |
            | A | A | B | A |   |   |   |   |

        Whereas otherwise, no realignment would be possible.

        :param nb_spacers: Number of empty events to insert between two events.
        :param update_window:
        """
        if self._warped_series is None:
            self._warped_series = self.series
        new_nb_events = (1+nb_spacers)*(self.nb_events-1)+1
        ws = np.zeros((self.nb_series, new_nb_events, self.nb_symbols), dtype=int)
        ws[:, 0, :] = self._warped_series[:, 0, :]
        for i in range(1, self.nb_events):
            ws[:, (1+nb_spacers)*i, :] = self._warped_series[:, i, :]
        self.series = ws
        self.nb_events = new_nb_events
        if update_window:
            self.window.insert_spacers(nb_spacers)
        self.reset()

    def get_counts(self, ignore_merged=False, filter_symbols=None, filter_series=None):
        if self._warped_series is None:
            self._warped_series = self.series
        if filter_symbols is None:
            ws = self._warped_series
        else:
            idxs, _, _ = np.where(self._warped_series[:, :, filter_symbols] > 0)
            ws = self._warped_series[idxs]
        if filter_series is None:
            filter_series = [True] * self.nb_series
        if ignore_merged:
            cnts = np.sign(ws)[filter_series].sum(axis=0).T
        else:
            cnts = ws[filter_series].sum(axis=0).T
        return cnts

    def get_smoothed_counts(self, window=3, ignore_merged=False, filter_symbols=None, filter_series=None):
        if self._warped_series is None:
            self._warped_series = self.series
        if filter_symbols is None:
            ws = self._warped_series
        else:
            idxs, _, _ = np.where(self._warped_series[:, :, filter_symbols] > 0)
            ws = self._warped_series[idxs]
        if filter_series is None:
            filter_series = [True] * self.nb_series
        if ignore_merged:
            counts = np.sign(ws)[filter_series]
        else:
            counts = ws[filter_series]
        kernel = signal.windows.hann(window)  # make sure to be uneven to be centered
        counts = counts.sum(axis=0).T
        for si in range(counts.shape[0]):
            counts[si, :] = signal.convolve(counts[si, :], kernel, mode='same') / sum(kernel)
        return counts

    def compute_windowed_counts(self):
        """Count over window and series

        Computes the following properties:
        - windowed_counts
        - factors

        Requires:
        - warped_series (or series)

        :return: Counts (also stored in self.windowed_counts)
        """
        if self._warped_series is None:
            self._warped_series = self.series
        window = self.window.counting(self._versions[V_WC])
        # Only count occurrence of a symbol in a series once, even though
        # we keep track of how many are merged but this information should not be used for counts
        ws = np.sign(self._warped_series)

        # Weigh based on number of occurrences per series (and symbol). Symbols
        # that occur in almost every timestep, should not impact the warping too much.
        max_entropy = np.log2(1 / self.nb_events)
        ws_sum = np.sum(ws, axis=1)
        ws_sum[ws_sum == 0] = 1
        self._factors = 1 - np.log2(np.divide(1, ws_sum)) / max_entropy
        self._factors[np.isinf(self._factors)] = 1
        ws = ws * self._factors[:, np.newaxis, :]
        self.ws = ws
        c = ws.sum(axis=0)

        # Sliding window to aggregate from neighbours
        if window > 1:
            sides = int((window - 1) / 2)
            wc = np.zeros((self.nb_symbols, self.nb_events))
            w = np.lib.stride_tricks.sliding_window_view(c, (window,), axis=(0,))
            wc[:, sides:-sides] = w.sum(axis=2).T
            # Pad the beginning and ending with the same values (having half a window can lower the count)
            wc[:, :sides] = wc[:, sides:sides+1]
            wc[:, -sides:] = wc[:, -sides-1:-sides]
            # Add without a window (otherwise the begin and end cannot differentiate)
            self._windowed_counts = np.add(wc, c.T) / 2
        else:
            self._windowed_counts = c.T

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

        Computes the following properties:
        - rescaled_counts

        Requires:
        - windowed_counts
        - factors

        """
        if self._versions[V_RC] == self._versions[V_WC]:
            self.compute_windowed_counts()
        if not self.rescale_active:
            self._rescaled_counts = self._windowed_counts
            return self._rescaled_counts

        # Normalize per symbol and reweigh with the factors
        countsp = np.power(self.windowed_counts, self.rescale_power)
        sums = countsp.sum(axis=1)
        sums[sums == 0.0] = 1.0
        countsp = countsp / sums[:, np.newaxis]
        factors = self._factors.sum(axis=0)  # sum per symbol
        partfn = np.sign(self._factors).sum(axis=0)  # nb of nonempty series per symbol
        factors = np.divide(factors, partfn)  # compute average factor per symbol over all series
        countsp = countsp * factors[:, np.newaxis]

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

    def update_gradients_without_warping(self, window=None, plot=None):
        """Recompute the gradients, possibly with a different window, without changing the series."""
        if plot is not None:
            import matplotlib.pyplot as plt
        else:
            plt = None
        if window is not None:
            self.window = window
        self.compute_rewards()
        self._versions[V_WS] += 1  # Do not change warping
        if plot is not None:
            fig, axs = self.plot_rewards(symbol=plot.get("symbol", None),
                                         seriesidx=plot.get("seriesidx", None))
            fig.savefig(plot['filename'], bbox_inches='tight')
            plt.close(fig)

    def compute_gradients(self):
        """
        Gradient of density for each item

        We use the windowed counts as the density function and use the difference as gradient

        Computes the following properties:
        - gradients
        - smoothed_counts (only used for plotting/debugging)

        Requires:
        - rescaled_counts

        :return: warping gradients (also stored in self.gradients)
        """
        # If windowed_counts has not yet been recomputed, do so
        if self._versions[V_WD] == self._versions[V_RC]:
            self.compute_rescaled_counts()
        window = self.window.smoothing(self._versions[V_WD])

        # Setup up kernel
        # Smooth window to double its size, a window on each side of the window
        # (but make sure it is uneven to be centered)
        kernel_side = window // 2
        kernel_width = int(kernel_side * 2 + 1)
        kernel = signal.windows.hann(kernel_width)  # make sure to be uneven to be centered

        # Convolve kernel
        counts = self._rescaled_counts
        countsf = np.zeros(counts.shape)
        for si in range(countsf.shape[0]):
            countsf[si, :] = signal.convolve(counts[si, :], kernel, mode='same') / sum(kernel)
        self._smoothed_counts = countsf

        gradients = np.diff(countsf)
        self.gradients = np.round(gradients, int(10 - np.floor(
            np.log10(np.max(np.abs(gradients))))))  # round 10 orders of magnitude
        return self.gradients


    def compute_rewards(self):
        """
        Computes forward and backward rewards for each item.

        Computes the following properties:
        - reward_forward
        - reward_backward

        Requires:
        - rescaled_counts

        :return: reward_forward, reward_backward
        """
        self.compute_gradients()

        self.reward_forward = np.zeros((self.nb_symbols, self.nb_events))
        self.reward_forward[:,:-1] = self.gradients
        self.reward_backward = np.zeros((self.nb_symbols, self.nb_events))
        self.reward_backward[:,1:] = - self.gradients

        # Increase backward reward of all (but the first) items in a flat density peak
        conti = np.where(self.reward_backward == 0.0)
        lastr, lastc = -1, -1
        for r, c in zip(*conti):
            if r == lastr and c <= lastc:
                continue
            if c == 0 or c == self.nb_events - 1:
                continue
            lastc = c + 1
            while lastc < self.nb_events and self.reward_backward[r, lastc] == 0:
                lastr = r
                lastc += 1
            if self.reward_backward[r, c - 1] < 0 < self.reward_backward[r, lastc]:
                epsilon = np.min(np.abs(self.gradients[self.gradients!=0]))
                self.reward_backward[r, (c+1):lastc] += epsilon

        self._versions[V_WD] += 1
        return self.reward_forward, self.reward_backward

    def get_symbol_information(self):
        """
        Compute 1-entropy/max_entropy of smoothed counts per symbol

        :return:
        """
        max_entropy = np.log(1 / self.nb_events)
        # -sum p(x) log(p(x))
        # Make densities from smoothed counts
        part = np.sum(self._smoothed_counts, axis=1)
        part[part == 0] = 1
        dens = self._smoothed_counts / part[:, np.newaxis]
        entr = special.entr(dens).sum(axis=1)
        # entr = - np.sum(np.multiply(dens, np.log2(dens)), axis=1)
        return 1 - entr / max_entropy

    @staticmethod
    def _best_warped_path(cc, ps=None):
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

        if ps is None:
            raise ValueError("Warping v2 requires a ps argument")
        for r in reversed(range(cc.shape[0] - 1)):
            prev_c = ps[r+1, prev_c]
            if prev_c == -1:
                return None
            path.append((r, r + prev_c - 1))
        path.reverse()
        return path

    def _allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        if self.constraints is None:
            return True
        for constraint in self.constraints:
            allowed = constraint.allow_merge(merged_cnts_s, merged_cnts_e, a, b)
            if not allowed:
                return False
        return True

    def compute_warped_series(self):
        """Warp events by maximally one step (left, right, or none).

        Computes the following properties:
        - warped_series

        Requires:
        - forward and backward rewards

        """
        # If rewards have not yet been recomputed, do so
        if self._versions[V_WD] == self._versions[V_WS]:
            self.compute_rewards()
        if self._warped_series_ec is None:
            self._warped_series_ec = self._warped_series.max(axis=2)

        assert self._warped_series.shape[0] == self.nb_series
        ws, wsec, converged, costs = self.align_series(self._warped_series, self._warped_series_ec)

        self._converged = None if converged is False else self._versions[V_WS]
        self._warped_series = ws
        self._warped_series_ec = wsec
        self._versions[V_WS] += 1
        self.costs += [costs]
        return self._warped_series


    def align_series(self, series, series_ec=None):
        """Align events in the given series based on the previously computed gradients.

        This method can be used to align new data to the originally given data. Assuming
        that warping has been applied to that originally given data.

        :param series: Series is an array with dimensions (series, event, symbol)
        :param series_ec: For every event, of how many merges this event is constructed.
            Dimensions are (series, nb of events)
        """
        if series_ec is None:
            series_ec = series.max(axis=2)
        assert series.shape[1] == self.nb_events, f"Expected series of length {series.shape[1]}, got {self.nb_events}"
        assert series.shape[2] == self.nb_symbols
        nb_series = series.shape[0]
        converged = True

        ws = np.zeros((nb_series, self.nb_events, self.nb_symbols), dtype=int)
        wsec = np.zeros((nb_series, self.nb_events), dtype=int)

        # Dynamic programming with window size 3. We only allow a shift of one or zero.
        cc = np.zeros((self.nb_events, 3))  # cumulative cost
        ps = np.zeros((self.nb_events, 3), dtype=int)  # previous state
        scs = np.zeros((self.nb_events, 3, self.nb_symbols), dtype=int)  # summed counts symbols
        sce = np.zeros((self.nb_events, 3), dtype=int)  # summed counts events

        # TODO: still written as costs instead of rewards:
        # Aggregated costs
        wss_all_backward = - np.einsum('kji,ij->kj', series, self.reward_backward)
        wss_all_forward = - np.einsum('kji,ij->kj', series, self.reward_forward)

        total_costs = np.zeros(nb_series)
        attr_costs = np.zeros(nb_series)
        for sei in range(nb_series):
            wssb = wss_all_backward[sei]
            wssf = wss_all_forward[sei]

            # Initialize datastructures
            cc[:, :] = 0
            ps[:, :] = 0
            scs[:, :, :] = 0
            sce[:, :] = 0

            # Initialize first row
            cc[0, 0] = np.inf   # First element cannot move backward
            cc[0, 1] = 0
            cc[0, 2] = wssf[0]
            ps[0, :] = [0, 1, 2]
            scs[0, 0, :] = 0
            scs[0, 1, :] = series[sei, 0, :]
            scs[0, 2, :] = scs[0, 1, :]
            sce[0, 0] = 0
            sce[0, 1] = series_ec[sei, 0]
            sce[0, 2] = sce[0, 1]
            for i in range(1, len(wssb)):
                # Backward
                #              Stay one behind
                #              |           Move on back from diagonal
                #              |           |             Cost to move backward
                # cc[i, 0] = min(cc[i-1, 0], cc[i-1, 1]) + wss[i]
                cc[i, 0] = np.inf
                ps[i, 1] = -1
                for prevs in [1, 0]:
                    if not cc[i - 1, prevs] < cc[i, 0]:
                        continue
                    if prevs == 1:
                        merged_cnts_s = scs[i-1, prevs] + series[sei, i, :]
                        merged_cnts_e = sce[i-1, prevs] + series_ec[sei, i]
                        args = scs[i-1, prevs], series[sei, i, :]
                    else:
                        merged_cnts_s = series[sei, i, :]
                        merged_cnts_e = series_ec[sei, i]
                        args = None, series[sei, i, :]
                    if self._allow_merge(merged_cnts_s, merged_cnts_e, *args):
                        cc[i, 0] = cc[i - 1, prevs]
                        ps[i, 0] = prevs
                        scs[i, 0] = merged_cnts_s
                        sce[i, 0] = merged_cnts_e
                cc[i, 0] += wssb[i]
                # Stay
                #              Move back to diagonal from one behind
                #              |           Stay on diagonal
                #              |           |           Move to diagonal from one ahead (thus stay)
                # cc[i, 1] = min(cc[i-1, 0], cc[i-1, 1], cc[i-1, 2])
                cc[i, 1] = np.inf
                ps[i, 1] = -1
                for prevs in [1, 0, 2]:
                    if not cc[i - 1, prevs] < cc[i, 1]:
                        continue
                    if prevs == 2:
                        merged_cnts_s = scs[i - 1, prevs] + series[sei, i, :]
                        merged_cnts_e = sce[i - 1, prevs] + series_ec[sei, i]
                        args = scs[i - 1, prevs], series[sei, i, :]
                    else:
                        merged_cnts_s = series[sei, i, :]
                        merged_cnts_e = series_ec[sei, i]
                        args = None, series[sei, i, :]
                    if self._allow_merge(merged_cnts_s, merged_cnts_e, *args):
                        cc[i, 1] = cc[i - 1, prevs]
                        ps[i, 1] = prevs
                        scs[i, 1] = merged_cnts_s
                        sce[i, 1] = merged_cnts_e
                    cc[i, 1] += 0
                # Forward
                #              Skip diagonal and move from one back for previous to one ahead for this one
                #              |           Move one forward from diagonal
                #              |           |           Stay one forward
                #              |           |           |             Cost to move forward
                # cc[i, 2] = min(cc[i-1, 0], cc[i-1, 1], cc[i-1, 2]) + wssf[i]
                cc[i, 2] = np.inf
                ps[i, 2] = -1
                for prevs in [1, 0, 2]:
                    if cc[i - 1, prevs] < cc[i, 2]:
                        cc[i, 2] = cc[i - 1, prevs]
                        ps[i, 2] = prevs
                        scs[i, 2] = series[sei, i, :]
                        sce[i, 2] = series_ec[sei, i]
                cc[i, 2] += wssf[i]
            cc[len(wssb) - 1, 2] = np.inf  # Last element cannot move forward
            path = self._best_warped_path(cc, ps)
            if path is None:
                print(f"No path found for series {sei}")
                continue

            # Do realignment
            for i_from, i_to in path:
                if converged is True and i_to != i_from:
                    converged = False
                ws[sei, i_to, :] = ws[sei, i_to, :] + series[sei, i_from, :]
                wsec[sei, i_to] = wsec[sei, i_to] + series_ec[sei, i_from]

            total_costs[sei] = np.min(cc[len(wssb) - 1])
            attr_costs[sei] = -np.sum(np.abs(wssb) * (np.array(path)[:,1]!=np.arange(1,len(path)+1)))
        total_costs = np.sum(total_costs)
        attr_costs = np.sum(attr_costs)
        dist = np.mean(np.sign(ws), axis=0) + 10**(-20)
        entropy2 = -np.sum(dist*np.log(dist))
        dist = dist / np.sum(dist, axis=0)
        entropy = -np.sum(dist*np.log(dist))
        print([total_costs, attr_costs, entropy])

        return ws, wsec, converged, [total_costs, attr_costs, entropy, entropy2]

    def align_series_times(self, series, series_ec=None, iterations=1):
        """Align events in the given series based on the previously computed gradients.

        This method can be used to align new data to the originally given data. Assuming
        that warping has been applied to that originally given data.

        See align_series for more information.
        """
        converged = None
        for idx in range(iterations):
            series, series_ec, converged, total_costs_it = self.align_series(series, series_ec)
            if converged is True:
                converged = idx
                break
            else:
                converged = None
        return series, series_ec, converged

    @property
    def warped_series(self):
        if self._warped_series is not None:
            return self._warped_series
        self._warped_series = self.series
        return self._warped_series

    @property
    def isconverged(self):
        if self._converged is None:
            return False
        else:
            return True

    @isconverged.setter
    def isconverged(self, value):
        if value is True:
            self._converged = -1
        elif type(value) is int:
            self._converged = value
        else:
            self._converged = None

    def compute_likelihoods(self, laplace_smoothing=1, exclude_items=(), selected_events = None):
        """Compute all the p(s_{i,j}|e_i).
        """
        self._loglikelihoods_p = np.divide(np.add(np.sign(self._warped_series).sum(axis=0), laplace_smoothing),
                                           self.nb_series + 2*laplace_smoothing)
        self._loglikelihoods_n = np.log(1.0 - self._loglikelihoods_p)
        self._loglikelihoods_p = np.log(self._loglikelihoods_p)

        for item in exclude_items:
            if self.intonly:
                ind = item
            else:
                ind = self.symbol2int[item]
            self._loglikelihoods_n[:, ind] = 0
            self._loglikelihoods_p[:, ind] = 0

        if selected_events is not None:
            self._loglikelihoods_n[~selected_events] = 0
            self._loglikelihoods_p[~selected_events] = 0

        self._exclude_items_ll = exclude_items

    def likelihood_with_model(self, laplace_smoothing=1, exclude_items=(), selected_events=None):
        """Compute likelihood of this EventSeries given the stored model."""
        if self.model is None:
            raise ValueError(f"This EventSeries has no associated model, use the 'likelihood' method.")
        return self.likelihood(self.model, laplace_smoothing, exclude_items, selected_events)

    def likelihood(self, model: 'EventSeries' = None, laplace_smoothing=1, exclude_items=(), selected_events=None):
        """Likelihood of the current eventseries given the 'model' eventseries.

        LL = p(x|M) = prod_i p(x_i|e_i)p(e_i|M) = prod_i p(x_i|e_i)
        with i the event index of the selected events
        All transitions and thus events are equally likely (this ignored).

        x is a set of symbols that can appear:
        p(x_i | e_i) = prod_j p(s_{i,j} | e_i)
        with j the symbol index

        LL = prod_{i,j} p(s_{i,j}|e_i)
        LLL = sum_{i,j} log(p(s_{i,j}|e_i))

        The intuition is that this is Naive Bayes for an event. What is the probability
        that it is the event at time i given the set of observations.
        Thus p(e_i | S) ~= prod_j p(s_j | e_i)

        :param model: EventSeries representing the model, this eventseries if None
        :return: Log Likelihood
        """
        if model is None:
            model = self
        if model._loglikelihoods_p is None or model._loglikelihoods_n is None or model._exclude_items_ll != exclude_items:
            model.compute_likelihoods(laplace_smoothing, exclude_items, selected_events)
        # TODO: check what's faster, this multiplies a lot of zeros, and requires np.sign
        ws = np.sign(self._warped_series)
        lll = np.einsum('ijk,jk->i',  ws, model._loglikelihoods_p)
        lll += np.einsum('ijk,jk->i',  1-ws, model._loglikelihoods_n)
        return lll

    def print_series(self):
        print(self.series)
        print(self.symbol2int)

    def format_series(self):
        return self._format_series(self.series)

    def format_warped_series(self, compact=False, drop_empty=False, drop_separators=False):
        if self._warped_series is None:
            raise ValueError(f"No warped series computed yet (use format_series).")
        return self._format_series(self._warped_series, compact=compact, drop_empty=drop_empty,
                                   drop_separators=drop_separators)

    def _format_series(self, series, compact=False, drop_empty=False, drop_separators=False):
        # (nb_series, nb_events, nb_symbols)
        if self.intonly:
            if compact:
                sl = 0
            else:
                sl = math.floor(math.log(self.nb_symbols - 1, 10)) + 1 + 1

            def fmt_symbol(sy_i):
                return f"{sy_i:>{sl}}"
        else:
            if compact:
                sl = 0
            else:
                sl = max(len(str(k)) for k in self.symbol2int.keys()) + 1

            def fmt_symbol(sy_i):
                return f"{self.int2symbol[sy_i]:>{sl}}"
        if compact:
            empty = ""
            ws = np.sign(self._warped_series)
            event_len = ws.sum(axis=2).max(axis=0)
        else:
            empty = " " * sl
            event_len = None

        s = ''
        for sei in range(self.nb_series):
            for evi in range(self.nb_events):
                if drop_empty and event_len[evi] == 0:
                    continue
                event_s = ''
                for syi in range(self.nb_symbols):
                    if series[sei, evi, syi]:
                        event_s += fmt_symbol(syi)
                    else:
                        event_s += empty
                if compact:
                    if event_len[evi] == 0:
                        event_s += ' '
                    else:
                        event_s += ' '*(event_len[evi] - len(event_s))
                s += event_s
                if evi != self.nb_events - 1:
                    if drop_separators:
                        pass
                    elif compact:
                        s += "|"
                    else:
                        s += " |"
            s += "\n"
        return s

    def plot_rewards(self, symbol=0, seriesidx=None, filename=None):
        """Plot the currently computed properties used to guide the warping.

        :param symbol: Add plots for the given symbol or list of symbols
        :param seriesidx: Add plots for the given series index or list of series indices.
        :param filename: Plot directly to a file (if not given, returns fig, axs)
        :return: fig, axs (if filename is not given)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        if type(symbol) is int:
            symbol = [symbol]
        elif type(symbol) is not list and isinstance(symbol, Iterable):
            symbol = list(symbol)
        elif symbol is None:
            symbol = []
        symbol = [self.symbol2int.get(cursymbol, cursymbol) for cursymbol in symbol]
        if type(seriesidx) is int:
            seriesidx = [seriesidx]
        elif type(seriesidx) is not list and isinstance(seriesidx, Iterable):
            seriesidx = list(seriesidx)
        elif seriesidx is None:
            seriesidx = []
        nrows = 2
        # if self.rescale_active:
        nrows += 1
        if seriesidx is not None:
            nrows += 2*len(seriesidx)
            wssf, wssb = [], []
            for si in range(len(seriesidx)):
                wssf.append(np.multiply(self.warped_series[seriesidx[si], :, :].T, self.reward_forward).sum(axis=0))
                wssb.append(np.multiply(self.warped_series[seriesidx[si], :, :].T, self.reward_backward).sum(axis=0))
        else:
            wssf = None
            wssb = None

        fig, axs = plt.subplots(nrows=nrows, ncols=len(symbol), sharex=True, sharey='row',
                                figsize=(5*len(symbol), 4+len(seriesidx)))
        cnts = self.get_counts(ignore_merged=True)
        colors = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]

        # plots per item
        for curidx, cursymbol in enumerate(symbol):
            curcnts = cnts[cursymbol]

            # Counts
            axrow = 0
            ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
            ax.set_title(f"Symbol {cursymbol}: {self.int2symbol.get(cursymbol, cursymbol)}")
            ax.bar(list(range(len(curcnts))), curcnts, color=colors[0], label="Counts")
            ax.plot(self._windowed_counts[cursymbol], '-o', color=colors[1], label="Counts (windowed)")
            if curidx == 0:
                ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
            axrow += 1

            # Scaled counts
            ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
            ax.plot(self._rescaled_counts[cursymbol], '-o', color=colors[2], label="Rescaled counts")
            ax.plot(self._smoothed_counts[cursymbol], '-o', color=colors[3], label="Smoothed counts")
            if curidx == 0:
                ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
            axrow += 1

            # Gradients and rewards
            ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
            ax.axhline(y=0, color='r', linestyle=':', alpha=0.3)
            ax.plot(self.gradients[cursymbol], '-o', color=colors[4], label="Gradients")
            ax.plot(self.reward_forward[cursymbol], '-o', color=colors[4], label="Forward rewards")
            ax.plot(self.reward_forward[cursymbol], '-o', color=colors[4], label="Backward rewards")
            if curidx == 0:
                ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
            axrow += 1

            # Counts in given series
            if seriesidx is not None:
                for si in range(len(seriesidx)):
                    ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
                    ax.axhline(y=0, color='r', linestyle=':', alpha=0.3)
                    ax.plot(wssf[si], '-+', color=colors[4], label=f"Agg forward rewards for series {seriesidx[si]}")
                    ax.plot(wssb[si], '-+', color=colors[5], label=f"Agg backward rewards for series {seriesidx[si]}")
                    if curidx == 0:
                        ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
                    axrow += 1
                    ax = axs[axrow, curidx] if len(symbol) > 1 else axs[axrow]
                    cursymbol_cnts = self.warped_series[seriesidx[si], :, cursymbol]
                    ax.bar(range(len(cursymbol_cnts)), cursymbol_cnts, label=f"Counts for series {seriesidx[si]}")
                    if curidx == 0:
                        ax.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
                    axrow += 1

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig, axs

    def plot_symbols(self, filename=None, filter_symbols=None, filter_series=None, title=None):
        """Plot the counts of all symbols over all events (aggregated over the series).

        :param filename: Plot directly to a file
        :param filter_symbols:
        :returns: fig, axs if filename is not given
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))

        ax = axs[0]
        if filter_series is not None:
            im = self.series[filter_series].sum(axis=0).T
        else:
            im = self.series.sum(axis=0).T
        ax.imshow(im)
        ax.set_xlabel('Events')
        ax.set_ylabel('Symbol')
        if title:
            ax.set_title("no warping")

        ax = axs[1]
        im = self.get_counts(ignore_merged=True, filter_symbols=filter_symbols, filter_series=filter_series)
        ax.imshow(im)
        ax.set_xlabel('Events')
        ax.set_ylabel('Symbol')
        if title:
            ax.set_title(title)

        ax = axs[2]
        thr = np.quantile(im, 0.9)
        im[im < thr] = 0
        ax.imshow(im)
        ax.set_xlabel('Events')
        ax.set_ylabel('Symbol')

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig, axs


    def plot_series(self, series_nr, filename=None):
        """Plot the counts of all symbols over all events of a series and the warped series

        :param filename: Plot directly to a file
        :param series_nr: which series to plot
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

        ax = axs[0]
        ax.imshow(self.series[series_nr].T)
        ax.set_xlabel('Events')
        ax.set_ylabel('Symbol')
        ax.set_title('series')

        ax = axs[1]
        ax.imshow(np.sign(self.warped_series[series_nr].T))
        ax.set_xlabel('Events')
        ax.set_ylabel('Symbol')
        ax.set_title('warped series')

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig, axs
