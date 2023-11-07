import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os

from eventwarping.eventseries import EventSeries


# If environment variable TESTDIR is set, save figures to this
# directory, otherwise to the test directory
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))


def test_example1():
    print("")
    fn = Path(__file__).parent / "rsrc" / "example1.txt"
    es = EventSeries.from_file(fn, window=3)
    # es.print_series()
    # print(es.series.shape)

    print("Series:")
    print(es.format_series())


    es.compute_windowed_counts()
    print(f"Windowed counts:\n{es.windowed_counts}")
    wc_sol = np.array([[4.5, 3.0, 1., 2.0, 3.0, 2., 1., 1.5],
                       [2.0, 2.5, 4., 2.5, 0.5, 0., 0., 0.0]])
    np.testing.assert_array_almost_equal(es.windowed_counts, wc_sol)

    es.compute_warping_directions()
    print(f"Warping directions:\n{es.warping_directions}")
    # wd_sol = np.array([[-1.8, -2.1, -0.6,  1.2,  0.0, -1.2, -0.3,  0.6],
    #                    [ 0.6,  1.2,  0.0, -2.1, -1.5, -0.3,  0.0,  0.0]])
    # np.testing.assert_array_almost_equal(es.warping_directions, wd_sol)

    es.compute_warped_series()
    print(f"Warped series:")
    print(es.format_warped_series())
    ws_sol = (" A   |     |   B |     | A   |     |     |     |\n"
              " A B |     |     |     | A   |     |     |     |\n"
              " A   |     |   B |     | A   |     |     |     |\n"
              " A   |     |   B |     | A   |     |     |     |\n"
              " A   |     |   B |     |     |     |     | A   |\n")
    np.equal(es.format_warped_series(), ws_sol)


def test_example2():
    print("")
    fn = Path(__file__).parent / "rsrc" / "example1.txt"
    es = EventSeries.from_file(fn, window=3)

    for i, ws in enumerate(es.warp_yield(iterations=3)):
        print(f"=== {i+1:>2} ===")
        print(es.format_warped_series())


def test_example3():
    print("")
    fn = Path(__file__).parent / "rsrc" / "example3.txt"
    es = EventSeries.from_file(fn, window=3)

    print("=== 1 ===")
    es.compute_windowed_counts()
    es.compute_warping_directions()
    print(es.format_warped_series())
    fig, axs = es.plot_directions(symbol=0)
    fig.savefig(directory / "gradients.png")
    es.compute_warped_series()

    print("=== 2 ===")
    es.warp()
    print(es.format_warped_series())


def test_example4():
    print("")
    fn = Path(__file__).parent / "rsrc" / "example4.txt"
    es = EventSeries.from_file(fn, window=3)

    print("=== 0 ===")
    es.compute_windowed_counts()
    es.compute_warping_directions()
    print(es.format_warped_series())
    fig, axs = es.plot_directions(symbol={0, 1, 2, 3})
    fig.savefig(directory / "gradients.png")
    plt.close(fig)
    es.compute_warped_series()

    print("=== 1 ===")
    print(es.format_warped_series())

    ws_sol = (" A B     |         |         |     C D |\n"
              " A B     |         |         |     C D |")
    np.equal(es.format_warped_series(), ws_sol)


def test_example5():
    print("")
    fn = Path(__file__).parent / "rsrc" / "example5.txt"
    es = EventSeries.from_file(fn, window=3)

    print("=== 0 ===")
    es.compute_windowed_counts()
    es.compute_warping_directions()
    print(es.format_warped_series())
    fig, axs = es.plot_directions(symbol={0, 1, 2, 3})
    fig.savefig(directory / "gradients.png")
    plt.close(fig)
    es.compute_warped_series()

    print("=== 1 ===")
    print(es.format_warped_series())
