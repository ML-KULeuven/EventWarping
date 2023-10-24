from pathlib import Path
import ast


def setlistfile2setlistsfile(fn_from, fn_to, start, stop, margin):
    sls = setlistfile2setlists(fn_from, start, stop, margin)
    if type(fn_to) is str:
        fn_to = Path(fn_to)
    with fn_to.open("w") as fp:
        for line in sls:
            fp.write(repr(line) + "\n")


def setlistfile2setlists(fn, start, stop, margin):
    """Translate one sequence of sets into a number of
    sequences based on begin and end symbols.

    :param fn: Filename or Path object
    :param start: Set of start symbols
    :param stop: Set of stop symbols
    :param margin: Number of sets to include before the start
        and after the stop
    :return: List[List[Set]]
    """
    if type(fn) is str:
        fn = Path(fn)
    with fn.open("r") as fp:
        data = fp.read()
    data = ast.literal_eval(data)
    return setlist2setlists(data, start, stop, margin)


def setlist2setlists(sl, start, stop, margin):
    starts, stops = [], []
    for i, s in enumerate(sl):
        if not start.isdisjoint(s):
            starts.append(i)
        if not stop.isdisjoint(s):
            stops.append(i)
    series = []
    for start, stop in zip(starts, stops):
        series.append(sl[start - margin:stop + 1 + margin])
    return series
