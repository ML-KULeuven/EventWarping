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


def test_likelihood7():
    """Warping leads to

         |     | B   |     |   A |   A |     |
         |   A | B   |     |     |   A |   A |
         |   A | B   |     |     |   A |     |

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

    new_es = EventSeries.from_string(
        " |   A | B   |     |     |   A |     | \n"
        " |   A | B   |     |   A |   A |     | \n"
        " |     |   A |   A |   A |   A |     | \n"
        " | B   |   A | B   |     | B   |     |   ",
        using=es)
    new_es.warp_with(es)

    print("")
    print(new_es.format_warped_series())

    es.compute_likelihoods(laplace_smoothing=0.1)
    llls = new_es.likelihood(model=es)
    for idx, lll in enumerate(llls):
        print(f'Likelihood[{idx}] = exp({lll:6.2f}) = {np.exp(lll):.5f}')

