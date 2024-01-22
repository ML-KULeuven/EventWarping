import math


class Window:
    def __init__(self, value):
        if type(value) is int:
            if value % 2 == 0:
                raise ValueError(f"Argument window should be an uneven number (got {value}).")
        elif type(value) is list:
            if any(value_i % 2 == 0 for value_i in value):
                raise ValueError(f"Argument window should be an uneven number (got {value}).")
        self._values = value

    def counting(self, item):
        return self._values

    def smoothing(self, item):
        return self._values

    def insert_spacers(self, nb_spacers):
        self._values = ((self._values // 2) + nb_spacers)*2 + 1

    def __str__(self):
        return str(self._values)

    @classmethod
    def wrap(cls, object):
        if isinstance(object, Window):
            return object
        if type(object) is int:
            return Window(object)
        raise ValueError(f"Unknown value type for a window: {object} ({type(object)}")


class LinearScalingWindow(Window):
    def __init__(self, start, stop=0, step=-2, delay=1):
        """Change the window size over time.

        :param start: Start with this window size (needs to be an uneven number)
        :param stop: Stop before reaching this window size
        :param step: Step to reduce the window size (needs to be an even number)
        :param delay: Keep the current window size for 'delay' number of times.
        """
        if start % 2 == 0:
            raise ValueError(f"Argument window.start should be an uneven number.")
        if step % 2 != 0:
            # Check that the step is even (to only have uneven values in the list)
            raise ValueError(f"Argument window.step should be an even number.")
        self._start = start
        self._stop = stop
        self._step = step
        self._delay = delay
        super().__init__(list(range(start, stop, step)))

    def counting(self, item):
        item = item // self._delay
        if item >= len(self._values):
            item = len(self._values) - 1
        return self._values.__getitem__(item)

    def smoothing(self, item):
        return self._start

    def insert_spacers(self, nb_spacers):
        new_start = ((self._values // 2) + nb_spacers)*2 + 1
        if new_start % 2 == 0:
            raise ValueError(f"Argument new_start should be an uneven number.")
        factor = math.floor(new_start / self._start)
        self._start = new_start
        self._step *= factor
        if self._step % 2 != 0:
            # Make sure the step is even (to only have uneven values in the list)
            self._step += 1
        self._values = list(range(self._start, self._stop, self._step))
