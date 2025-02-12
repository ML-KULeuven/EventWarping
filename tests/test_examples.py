from typing import Iterable

import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os

from eventwarping.eventseries import EventSeries
from eventwarping.constraints import *
from eventwarping.window import StaticWindow, MultipleWindow

from src.eventwarping.window import Window, LinearScalingWindow

# If environment variable TESTDIR is set, save figures to this
# directory, otherwise to the test directory
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
doplots = True #if os.environ.get('TESTPLOTS', 0) in [1, '1'] else False


def test_example1():
    fn = Path(__file__).parent / "rsrc" / "example1.txt"
    es = EventSeries.from_file(fn, window=3)

    es.compute_windowed_counts()

    es.compute_rewards()

    es.compute_warped_series()
    ws_sol = (" A   |     |   B |     | A   |     |     |     |\n"
              " A B |     |     |     | A   |     |     |     |\n"
              " A   |     |   B |     | A   |     |     |     |\n"
              " A   |     |   B |     | A   |     |     |     |\n"
              " A   |     |   B |     |     |     |     | A   |\n")
    np.equal(es.format_warped_series(), ws_sol)


def test_example2():
    fn = Path(__file__).parent / "rsrc" / "example1.txt"
    es = EventSeries.from_file(fn, window=3)

    for i, ws in enumerate(es.warp_yield(iterations=3)):
        pass

    ws_sol = (" A   |     |   B |     | A   |     |     |    \n"
              "     |     | A B |     | A   |     |     |    \n"
              " A   |     |   B |     | A   |     |     |    \n"
              " A   |     |   B |     | A   |     |     |    \n"
              " A   |     |   B |     |     |     |     | A  \n")
    np.equal(es.format_warped_series(), ws_sol)


def test_example3():
    fn = Path(__file__).parent / "rsrc" / "example3.txt"
    es = EventSeries.from_file(fn, window=3)

    for i in range(2):
        plot = None
        es.warp(plot=plot)

    ws_sol = (" A |   |   |   | A |   |   |  \n"
              " A |   |   |   | A |   |   |  \n"
              " A |   |   |   | A |   |   |  \n"
              " A |   |   |   | A |   |   |  \n"
              " A |   |   |   |   |   |   | A\n")
    np.equal(es.format_warped_series(), ws_sol)


def test_example4():
    fn = Path(__file__).parent / "rsrc" / "example4.txt"
    es = EventSeries.from_file(fn, window=3)

    for i in range(2):
        plot = None
        es.warp(plot=plot)

    ws_sol = (" A B     |         |         |     C D |\n"
              " A B     |         |         |     C D |")
    np.equal(es.format_warped_series(), ws_sol)


def test_example5():
    fn = Path(__file__).parent / "rsrc" / "example5.txt"
    es = EventSeries.from_file(fn, window=3)

    for i in range(3):
        plot = None
        es.warp(plot=plot)

    ws_sol = (" A C B   |         |         |   C   D\n",
              " A   B   |         |         |   C   D")
    np.equal(es.format_warped_series(), ws_sol)


def test_example6():
    fn = Path(__file__).parent / "rsrc" / "example6.txt"
    es = EventSeries.from_file(fn, window=3, constraints=[MaxMergeEventConstraint(2)])

    for i in range(10):
        es.warp()
    ws_sol = ("   |   |   | A |   |   |   |  \n"
              " A |   | A | A | A | A |   | A")
    np.equal(es.format_warped_series(), ws_sol)

    es = EventSeries.from_file(fn, window=3)
    for i in range(10):
        es.warp()
    ws_sol = ("   |   |   | A |   |   |   |  "
              " A |   |   | A |   |   |   | A")
    np.equal(es.format_warped_series(), ws_sol)


def test_example7():
    """To check:

    - Are the Bs aligned
    - For the As to be aligned at the end, see test_example7b

    """
    fn = Path(__file__).parent / "rsrc" / "example7.txt"
    es = EventSeries.from_file(fn, window=5, constraints=[MaxMergeEventConstraint(1)])

    for i in range(3):
        plot = None
        es.warp(plot=plot)

    ws_sol = ("     |     | B   |     |   A |   A |     |    "
              "     |   A | B   |     |     |   A |   A |    "
              "     |   A | B   |     |     |   A |     |   ")
    np.equal(es.format_warped_series(), ws_sol)


def test_example7b():
    """To check:

    - Are the Bs aligned
    - Are the As aligned in the end

    This could be suboptimal
        |     | B   |     |   A |   A |     |
        |   A | B   |     |     |   A |   A |
    Thus prefer (by using different constraint):
        |     | B   |     |     |   A |     |
        |   A | B   |     |     |   A |     |

    """
    fn = Path(__file__).parent / "rsrc" / "example7.txt"
    es = EventSeries.from_file(fn, window=5, constraints=[MaxMergeEventIfSameConstraint(2)])

    for i in range(3):
        es.warp()

    ws_sol = ("     |     | B   |     |     |   A |     |    "
              "     |   A | B   |     |     |   A |     |    "
              "     |   A | B   |     |     |   A |     |    ")
    np.equal(es.format_warped_series(), ws_sol)


def test_example8():
    fn = Path(__file__).parent / "rsrc" / "example8.txt"
    es = EventSeries.from_file(fn, window=5, constraints=[MaxMergeEventConstraint(1)])
    es.insert_spacers(1)

    for i in range(5):
        plot = None
        es.warp(plot=plot)

    ws_sol = ("     |     |     |     | A   |   B |     | A   | A   | A   | A   | A   |    "
              "     | A   | A   | A   | A   |   B |     |     |     |     | A   | A   |    ")
    np.equal(es.format_warped_series(), ws_sol)


def test_example9():
    data = (
        "| A | A   C | B |   |\n"
        "| A |   B   | C | C |\n"
    )
    constraints = [
        MaxMergeEventConstraint(2),
        NoXorMergeSymbolSetConstraint(["A", "B"])
    ]
    es = EventSeries.from_string(data, window=5, constraints=constraints)

    for i in range(5):
        plot = None
        es.warp(plot=plot)

    ws_sol = ("       | A C   |     B |       |       |      "
              "       | A     |     B |   C   |       |      ")
    np.equal(es.format_warped_series(), ws_sol)


def test_example10():
    data = (
        "| A | A B | B C | C |\n"
        "| A |     |     | C |\n"
    )

    def distance(a, b):
        if ('B' in a and 'C' in b) or ('B' in b and 'C' in a):
            return 5
        return 1

    constraints = [
        NoMergeTooDistantSymbolSetConstraint(distance, 2)
    ]
    es = EventSeries.from_string(data, window=5, constraints=constraints)

    for i in range(5):
        es.warp()

    ws_sol = ("       | A B   |       |   B C |     C |      "
              "       | A     |       |       |     C |      ")
    np.equal(es.format_warped_series(), ws_sol)


def test_example11():
    data = ["PASPSLS",
            "SDLFPASTSLS",
            "PTSPSLS",
            "SDLFPTSTSLS"]

    window = MultipleWindow([5, 1], [13, 3], delay="convergence")
    es = EventSeries.from_chararrays(data, window=window, constraints=[MaxMergeEventConstraint(1)])
    es.insert_spacers(1, update_window=False)
    es.warp(iterations=20)

    es.insert_spacers(1, update_window=False)
    window = MultipleWindow([3, 1], [5, 3], delay="convergence")
    es.warp(iterations=5, restart=True, window=window)

    ws_res = es.format_warped_series(compact=True, drop_empty=True, drop_separators=True)
    ws_sol = ["    PA  SPSLS",
              "SDLFPASTS  LS",
              "    P  TSPSLS",
              "SDLFP  TSTSLS", ""]
    assert ws_res == "\n".join(ws_sol)


def test_example11b():
    """Inspired by protein sequence alignment.

    https://www.ebi.ac.uk/seqdb/confluence/display/JDSAT/Multiple+Sequence+Alignment+Tool+Input+Examples

    Problem:  The last symbols will not align even though there the same.
    This is because the same symbols reappear too often. Too many peaks.
    EventWarping anchors to symbols that are infrequent in a series, but frequent
    across series.
    Solution: Larger smoothing window than counting window
    """
    data = ["PSLS",
            "FPASTSLS",
            "PSLS",
            "FPTSTSLS"]

    es = EventSeries.from_chararrays(data, window=MultipleWindow([5, 1], [13, 5], delay="convergence"),
                                     constraints=[MaxMergeEventConstraint(1)])
    if doplots:
        plot = {'filename': str(directory / 'gradients_{iteration}.png'), 'symbol': {'S', 'L'}, 'seriesidx': (0, 1)}
    else:
        plot = None
    es.insert_spacers(1)
    es.warp(iterations=10, plot=plot)
    print(f"\nDone warping. {es.converged_str}")
    ws_res = es.format_warped_series(compact=True, drop_empty=True, drop_separators=True)

    ws_sol = [" P   SLS",
              "FPASTSLS",
              " P   SLS",
              "FPTSTSLS", ""]
    assert ws_res == "\n".join(ws_sol)


def test_inertia():
    fn = Path(__file__).parent / "rsrc" / "example_inertia.txt"
    es = EventSeries.from_file(fn, window=1)
    es.compute_windowed_counts()
    es.compute_rewards()

    assert np.all(np.sign(es.reward_backward) == [0, -1, 0, 1, 1, 1])
    assert np.all(np.sign(es.reward_forward) == [1, 0, 0, 0, -1, 0])
