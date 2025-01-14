from pathlib import Path
import ast


def setlistfile2setlistsfile(fn_from, fn_to, start, stop, margin, symbol_ordenings=None):
    sls = setlistfile2setlists(fn_from, start, stop, margin, symbol_ordenings)

    if type(fn_to) is str:
        fn_to = Path(fn_to)
    with fn_to.open("w") as fp:
        for line in sls:
            fp.write(repr(line) + "\n")


def setlistfile2setlists(fn, start, stop, margin, symbol_ordenings=None):
    """Translate one sequence of sets into a number of
    sequences based on begin and end symbols.

    :param fn: Filename or Path objectsetlistfile2setlistsfile
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
    return setlist2setlists(data, start, stop, margin, symbol_ordenings)


def setlist2setlists(sl, start, stop, margin, symbol_ordenings=None):
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

    if symbol_ordenings:
        series = smooth_series(series, symbol_ordenings)
    return series


def smooth_series(series, symbol_ordenings):
    """Modifies the series such that all intermediate values are included as well.
    The symbol partitioning indicates which symbols have an ordering

    e.g.,   symbol_partition = [[1,2,3]]
            series = [[{0,2},{5},{1},{},{3}]]
            outputs [[{0,2},{5},{1,2},{2},{3}]] as 2 should be between 1 and 3

    :param series: An event series
    :param symbol_ordenings: List of lists of ordered items
    :return: List[List[Set]]
    """
    for serie in series:
        for ordening in symbol_ordenings:
            a = [i.intersection(ordening) for i in serie]
            # add items per itemset
            for i,j in enumerate(a):
                if len(j) > 1:
                    serie[i] = serie[i].union(range(min(j), max(j)+1))
            # add items to make continuous
            prev = None
            for i, j in enumerate(a):
                if len(j) == 0:
                    continue
                if prev is None:
                    prev = [min(j), max(j), i]
                    continue

                new = [min(j), max(j), i]
                if new[0] > (prev[1]+1):
                    to_add = range(prev[1] + 1, new[0])
                    for k in range(prev[2], new[2]):
                        serie[k] = serie[k].union(to_add)
                if new[1] < (prev[0]-1):
                    to_add = range(new[1] + 1, prev[0])
                    for k in range(prev[2], new[2]+1):
                        serie[k] = serie[k].union(to_add)
                prev = [min(j), max(j), i]
    return series