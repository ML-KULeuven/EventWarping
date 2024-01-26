import math
import abc
from abc import ABC
from typing import List
from dataclasses import dataclass, field


class Window(ABC):
    def __init__(self, count_window, smooth_window = None):
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

    def __str__(self):
        return f"Window({self._count_window}, {self._smooth_window})"

    @classmethod
    def wrap(cls, object):
        if isinstance(object, Window):
            return object
        if type(object) is int:
            return StaticWindow(object)
        if type(object) in [list, tuple] and len(object) == 2:
            return StaticWindow(*object)
        if type(object) in [list, tuple] and 2 < len(object) <= 4:
            return LinearScalingWindow(*object)
        raise ValueError(f"Unknown value type for a window: {object} ({type(object)}")


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

    @staticmethod
    def _check_uneven(window, name):
        if type(window) is not int:
            raise ValueError(f"Argument {name} should be an integer (got {type(window)})")
        if window % 2 == 0:
            raise ValueError(f"Argument {name} should be an uneven number (got {window}).")

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


@dataclass
class ListRange:
    start: int
    stop: int
    step: int
    delay: int
    list: List[int] = field(init=False)

    def __post_init__(self):
        if self.start % 2 == 0:
            raise ValueError(f"Argument start={self.start} should be an uneven number.")
        if self.step % 2 != 0:
            # Check that the step is even (to only have uneven values in the list)
            raise ValueError(f"Argument step={self.step} should be an even number.")

        self.list = list(range(self.start, self.stop, self.step))
        if len(self.list) == 0:
            raise ValueError(f"The range is empty: ({self.start},{self.stop},{self.step})")

    def __getitem__(self, item):
        item = item // self.delay
        if item >= len(self.list):
            item = len(self.list) - 1
        return self.list.__getitem__(item)

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
        return ListRange(start, self.stop, step, self.delay)


class LinearScalingWindow(Window):
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
        delay_c, delay_s = self._parse_window(delay, 'delay')

        super().__init__(ListRange(start_c, stop_c, step_c, delay_c),
                         ListRange(start_s, stop_s, step_s, delay_s))

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

    def counting(self, item):
        return self._count_window.__getitem__(item)

    def smoothing(self, item):
        return self._smooth_window.__getitem__(item)

    def insert_spacers(self, nb_spacers):
        self._count_window = self._count_window.insert_spacers(nb_spacers)
        self._smooth_window = self._smooth_window.insert_spacers(nb_spacers)
