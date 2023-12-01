import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os

from eventwarping.eventseries import EventSeries
from eventwarping.constraints import *


# If environment variable TESTDIR is set, save figures to this
# directory, otherwise to the test directory
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))


def test_example1():
    fn = Path(__file__).parent / "rsrc" / "example1.txt"
    es = EventSeries.from_file(fn, window=3)
    # print("Series:\n{es.format_series()}")

    es.compute_windowed_counts()
    # print(f"Windowed counts:\n{es.windowed_counts}")
    # wc_sol = np.array([[4.5, 3.0, 1., 2.0, 3.0, 2., 1., 1.5],
    #                    [2.0, 2.5, 4., 2.5, 0.5, 0., 0., 0.0]])
    # np.testing.assert_array_almost_equal(es.windowed_counts, wc_sol)

    es.compute_warping_directions()
    # print(f"Warping directions:\n{es.warping_directions}")
    # wd_sol = np.array([[-1.8, -2.1, -0.6,  1.2,  0.0, -1.2, -0.3,  0.6],
    #                    [ 0.6,  1.2,  0.0, -2.1, -1.5, -0.3,  0.0,  0.0]])
    # np.testing.assert_array_almost_equal(es.warping_directions, wd_sol)

    es.compute_warped_series()
    # print(f"Warped series:\n{es.format_warped_series()}")
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
        # print(f"=== {i+1:>2} ===")
        # print(es.format_warped_series())
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

    # print(f'Original\n{es.format_series()}')
    for i in range(2):
        # print(f"=== {i} ===")
        # plot = {'filename': str(directory / f'gradients_{i}.png'), 'symbol': [0], 'seriesidx': []}
        plot = None
        es.warp(plot=plot)
        # print(es.format_warped_series())
    # print(f'Result:\n{es.format_warped_series()}')

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
        # print(f"=== {i} ===")
        # plot = {'filename': str(directory / f'gradients_{i}.png'), 'symbol': [0, 1, 2, 3], 'seriesidx': []}
        plot = None
        es.warp(plot=plot)
    # print(f'Result:\n{es.format_warped_series()}')

    ws_sol = (" A B     |         |         |     C D |\n"
              " A B     |         |         |     C D |")
    np.equal(es.format_warped_series(), ws_sol)


def test_example5():
    fn = Path(__file__).parent / "rsrc" / "example5.txt"
    es = EventSeries.from_file(fn, window=3)

    for i in range(3):
        # print(f"=== {i} ===")
        # plot = {'filename': str(directory / f'gradients_{i}.png'), 'symbol': [0, 1, 2, 3], 'seriesidx': []}
        plot = None
        es.warp(plot=plot)
    # print(f'Result:\n{es.format_warped_series()}')

    ws_sol = (" A C B   |         |         |   C   D\n",
              " A   B   |         |         |   C   D")
    np.equal(es.format_warped_series(), ws_sol)


def test_example6():
    fn = Path(__file__).parent / "rsrc" / "example6.txt"
    es = EventSeries.from_file(fn, window=3, constraints=[MaxMergeEventConstraint(2)])
    # print(f'Original:\n{es.format_series()}')

    for i in range(10):
        es.warp()
    # print(f'With constraints:\n{es.format_warped_series()}')
    ws_sol = ("   |   |   | A |   |   |   |  \n"
              " A |   | A | A | A | A |   | A")
    np.equal(es.format_warped_series(), ws_sol)

    es = EventSeries.from_file(fn, window=3)
    for i in range(10):
        es.warp()
    # print(f'Without constraints:\n{es.format_warped_series()}')
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
    # print('Original:\n{es.format_warped_series()}')

    for i in range(3):
        # print(f"=== {i} ===")
        # plot = {'filename': str(directory / f'gradients_{i}.png'), 'symbol': {0, 1}, 'seriesidx': 0}
        plot = None
        es.warp(plot=plot)
        # print(es.format_warped_series())
    # print(f'Result:\n{es.format_warped_series()}')

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
    # print(f'Original:\n{es.format_series()}')

    for i in range(3):
        # print(f"=== {i} ===")
        es.warp()
        # print(es.format_warped_series())
    # print(f'Result:\n{es.format_warped_series()}')

    ws_sol = ("     |     | B   |     |     |   A |     |    "
              "     |   A | B   |     |     |   A |     |    "
              "     |   A | B   |     |     |   A |     |    ")
    np.equal(es.format_warped_series(), ws_sol)


def test_example8():
    fn = Path(__file__).parent / "rsrc" / "example8.txt"
    es = EventSeries.from_file(fn, window=5, constraints=[MaxMergeEventConstraint(1)])
    # print(f'Original\n{es.format_series()}')
    es.insert_spacers(1)
    # print(f'Spaced\n{es.format_series()}')

    for i in range(5):
        # print(f"=== {i} ===")
        # plot = {'filename': str(directory / f'gradients_{i}.png'), 'symbol': {0, 1}, 'seriesidx': 1}
        plot = None
        es.warp(plot=plot)
        # print(es.format_warped_series())
    # print(f'Result:\n{es.format_warped_series()}')

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
    # print(f'Original\n{es.format_series()}')

    for i in range(5):
        # print(f"=== {i} ===")
        # plot = {'filename': str(directory / f'gradients_{i}.png'), 'symbol': [0, 1, 2], 'seriesidx': [0, 1]}
        plot = None
        es.warp(plot=plot)
        # print(es.format_warped_series())
    # print(f'Result:\n{es.format_warped_series()}')

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

