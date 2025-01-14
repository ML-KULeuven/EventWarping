import math
import abc
from abc import ABC
from typing import List
from dataclasses import dataclass, field


class Window(ABC):
    def __init__(self, count_window, smooth_window=None):
        """Create a window that has a different size for the window for
        computing counts and for smoothing the counts.

        :param count_window: The window in which events should be considered to be
            the same peak. Alternatively, the window over which the symbols should be considered
            to be at the same time (thus that could be the same event, but measurements
            are not synced perfectly).
        :param smooth_window: The window over which you want to do warping. Thus, the distance
            over which you want a peak to attract symbols.
            This is typically equal or larger than the count window.
            Note that a value of <=3 means no smoothing (a Hann window is used).
        """
        self._count_window = count_window
        self._smooth_window = smooth_window

    @staticmethod
    def _check_uneven(window, name):
        if type(window) is list:
            for w in window:
                Window._check_uneven(w, name+".item")
        elif type(window) is int:
            if window % 2 == 0:
                raise ValueError(f"Argument {name} should be an uneven number, got {window} ({type(window)}).")
        else:
            raise ValueError(f"Argument {name} should be an integer, got {window} ({type(window)})")

    @abc.abstractmethod
    def counting(self, item):
        """Window to use for counting.

        :param item: Current iteration number
        :returns: Window size
        """
        pass

    @abc.abstractmethod
    def smoothing(self, item):
        """Window to use for smoothing.

        :param item: Current iteration number
        :returns: Window size
        """
        pass

    @abc.abstractmethod
    def insert_spacers(self, nb_spacers):
        """Enlarge the window to take into account the spacers inserted.

        :param nb_spacers: Number of spacers
        """
        pass

    def next_window(self):
        return False

    def __len__(self):
        return 1

    def __str__(self):
        return f"Window({self._count_window}, {self._smooth_window})"

    @classmethod
    def wrap(cls, object):
        return object
        # if isinstance(object, Window):
        #     return object
        # if type(object) is int:
        #     return StaticWindow(object)
        # if type(object) in [list, tuple] and len(object) == 2:
        #     return StaticWindow(*object)
        # if type(object) in [list, tuple] and 2 < len(object) <= 4:
        #     return LinearScalingWindow(*object)
        # raise ValueError(f"Unknown value type for a window: {object} ({type(object)}")


class StaticWindow(Window):
    def __init__(self, count_window: int, smooth_window: int = None):
        """Create a window with a static size
        """
        self._check_uneven(count_window, 'count_window')
        if smooth_window is None:
            smooth_window = count_window
        else:
            self._check_uneven(smooth_window, 'smooth_window')
        super().__init__(count_window, smooth_window)

    def counting(self, item):
        """Window to use for counting.

        :param item: Current iteration number
        :returns: Window size
        """
        return self._count_window

    def smoothing(self, item):
        """Window to use for smoothing.

        :param item: Current iteration number
        :returns: Window size
        """
        return self._smooth_window

    def insert_spacers(self, nb_spacers):
        """Enlarge the window to take into account the spacers inserted.

        :param nb_spacers: Number of spacers
        """
        self._count_window = ((self._count_window // 2) + nb_spacers)*2 + 1
        self._smooth_window = ((self._smooth_window // 2) + nb_spacers) * 2 + 1

    def __str__(self):
        return f"StaticWindow({self._count_window}, {self._smooth_window})"


class MultipleWindow(Window):
    def __init__(self, count_window: List[int], smooth_window=None, delay=None):
        """Change the window size over time according to the given list.

        The current iteration is the list index used (or the max index).
        If delay is given an integer, the value in the given list is reused `delay` times.
        If delay is given a string 'convergence', the list index is the
        number of times convergence or the number of iterations is reached. In this case,
        the method does `len(count_window)*iterations` warping steps instead of `iterations`.

        :param count_window: List of count windows
        :param smooth_window: List of smoothing windows
        :param delay: Integer or 'convergence'
        """
        MultipleWindow._check_uneven(count_window, "count_window")
        if smooth_window is None:
            smooth_window = count_window
        else:
            MultipleWindow._check_uneven(smooth_window, "smooth_window")
        # assert len(count_window) == len(smooth_window)
        super().__init__(count_window, smooth_window)
        self.delay = delay
        if type(self.delay) == str and self.delay != "convergence":
            raise AttributeError(f"Unknown delay type: {self.delay}")
        self.nb_convergences = 0

    def counting(self, item):
        if type(self.delay) is int:
            item = item // self.delay
        elif self.delay == "convergence":
            item = self.nb_convergences
        if item >= len(self._count_window):
            item = len(self._count_window) - 1
        return self._count_window.__getitem__(item)

    def smoothing(self, item):
        if type(self.delay) is int:
            item = item // self.delay
        elif self.delay == "convergence":
            item = self.nb_convergences
        if item >= len(self._smooth_window):
            item = len(self._smooth_window) - 1
        return self._smooth_window.__getitem__(item)

    def insert_spacers(self, nb_spacers):
        self._count_window = [((w // 2) + nb_spacers)*2 + 1 for w in self._count_window]
        self._smooth_window = [((w // 2) + nb_spacers) * 2 + 1 for w in self._smooth_window]

    def next_window(self):
        if self.delay == "convergence":
            self.nb_convergences += 1
            if self.nb_convergences == self.__len__():
                return False
            return True
        return False

    def __len__(self):
        return max(len(self._count_window), len(self._smooth_window))


@dataclass
class ListRange:
    start: int
    stop: int
    step: int

    def list(self):
        return list(range(self.start, self.stop, self.step))

    def insert_spacers(self, nb_spacers):
        new_start = ((self.start // 2) + nb_spacers) * 2 + 1
        if new_start % 2 == 0:
            raise ValueError(f"Argument new_start should be an uneven number.")
        factor = math.floor(new_start / self.start)
        start = new_start
        step = self.step * factor
        if step % 2 != 0:
            # Make sure the step is even (to only have uneven values in the list)
            step += 1
        return ListRange(start, self.stop, step)


class LinearScalingWindow(MultipleWindow):
    def __init__(self, start, stop=0, step=-2, delay=1):
        """Change the counting window size over time. By default, the smoothing
        window is constant and set to the start value. If you want to deviate,
        provide a tuple with two values (first is counting, second is smoothing).

        :param start: Start with this window size (needs to be an uneven number)
        :param stop: Stop before reaching this window size
        :param step: Step to reduce the window size (needs to be an even number)
        :param delay: Keep the current window size for 'delay' number of times.
        """
        start_c, start_s = self._parse_window(start, 'start')
        stop_c, stop_s = self._parse_window(stop, 'stop', default=start-1)
        step_c, step_s = self._parse_window(step, 'step')

        self._count_listrange = ListRange(start_c, stop_c, step_c)
        self._smooth_listrange = ListRange(start_s, stop_s, step_s)

        super().__init__(self._count_listrange.list(), self._smooth_listrange.list(), delay=delay)

    @staticmethod
    def _parse_window(value, name='window', default=None):
        c, s = None, None
        if type(value) is int:
            c = value
            s = value if default is None else default
        elif type(value) in [list, tuple]:
            c, s = value
        else:
            raise AttributeError(f"Unknown type for {name}: {value} ({type(value)})")
        return c, s

    def insert_spacers(self, nb_spacers):
        self._count_listrange = self._count_listrange.insert_spacers(nb_spacers)
        self._smooth_listrange = self._smooth_listrange.insert_spacers(nb_spacers)

        self._count_window = self._count_listrange.list()
        self._smooth_window = self._smooth_listrange.list()
