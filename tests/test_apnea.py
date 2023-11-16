import pytest
import numpy as np
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from eventwarping.eventseries import EventSeries
from eventwarping.formats import setlistfile2setlistsfile

# If environment variable TESTDIR is set, save figures to this
# directory, otherwise to the test directory
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))


def get_data(fn):
    rsrc = Path(__file__).parent / "rsrc" / "apnea"
    path = rsrc / fn
    if not path.exists():
        fn2 = fn.replace("_series", "")
        path2 = rsrc / fn2
        print(f"Cannot find {path}. Trying to create from {path2}")
        if not path2.exists():
            raise Exception(f"Cannot find files: {path} or {path2}")
        setlistfile2setlistsfile(path2, path, start={1, 3}, stop={2, 4}, margin=3)
    return path


def test_example1():
    fn = get_data("MST_M03_0004086_itemsets_series.txt")
    es = EventSeries.from_setlistfile(fn, window=3, intonly=True)
    # Prefer to group initial / end states
    for symbol in [1, 2, 3, 4]:
        es.rescale_weights[symbol] = 10

    for i in range(3):
        print(f"=== {i+1:>2} ===")
        es.compute_warping_directions()
        fig, axs = es.plot_directions(symbol={3,4,5}, seriesidx=2)
        fig.savefig(directory / f"gradients_{i}.png", bbox_inches='tight')
        plt.close(fig)
        es.compute_warped_series()
        print(es.format_warped_series())


def test_example1_v2():
    fn = get_data("MST_M03_0004086_itemsets_series.txt")
    es = EventSeries.from_setlistfile(fn, window=5, intonly=True)
    es._use_warping_v2 = True
    es.allow_merge = 2
    # Prefer to group initial / end states
    for symbol in [1, 2, 3, 4]:
        es.rescale_weights[symbol] = 5

    for i in range(5):
        print(f"=== {i+1:>2} ===")
        es.compute_warping_directions()
        fig, axs = es.plot_directions(symbol={3,4,5}, seriesidx=2)
        fig.savefig(directory / f"gradients_{i}.png", bbox_inches='tight')
        plt.close(fig)
        es.compute_warped_series()
        print(es.format_warped_series())
