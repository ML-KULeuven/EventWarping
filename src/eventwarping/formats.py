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
    for startidx, stopidx in zip(starts, stops):
        # Avoid that the end of a previous segment is included in the margin before the current segment
        lmargin = 0
        for _ in range(margin):
            if startidx - lmargin - 1 < 0:
                break
            if not stop.isdisjoint(sl[startidx - lmargin - 1]):
                break
            lmargin += 1
        # Avoid that the start of a next segment is included in the margin after the current segment
        rmargin = 0
        for _ in range(margin):
            if stopidx + rmargin + 1 == len(sl):
                rmargin += 1
                break
            if not start.isdisjoint(sl[stopidx + rmargin + 1]):
                break
            rmargin += 1
        series.append(sl[startidx - lmargin:stopidx + 1 + rmargin])
    return series
