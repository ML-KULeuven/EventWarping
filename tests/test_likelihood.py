import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os

from eventwarping.eventseries import EventSeries
from eventwarping.constraints import *
from eventwarping.window import LinearScalingWindow, CountAndSmoothWindow


# If environment variable TESTDIR is set, save figures to this
# directory, otherwise to the test directory
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))


def test_likelihood7():
    """Warping leads to

         |     | B   |     |   A |   A |     |
         |   A | B   |     |     |   A |   A |
         |   A | B   |     |     |   A |     |

    """
    use_scaling_window = True
    fn = Path(__file__).parent / "rsrc" / "example7.txt"
    if use_scaling_window:
        es = EventSeries.from_file(fn, window=LinearScalingWindow(5), constraints=[MaxMergeEventConstraint(1)])
    else:
        es = EventSeries.from_file(fn, window=5, constraints=[MaxMergeEventConstraint(1)])
    # print('Original:\n{es.format_warped_series()}')

    nb_iterations = 3
    plot = {'filename': str(directory / 'gradients_{iteration}.png'), 'symbol': {0, 1}} # , 'seriesidx': 1}
    # plot = None
    es.warp(iterations=nb_iterations, plot=plot)

    if not use_scaling_window:
        # Do one additional iteration to update the gradients with a different window
        plot = {'filename': str(directory / f'gradients_{nb_iterations}.png'), 'symbol': {0, 1}}
        es.update_gradients_without_warping(window=CountAndSmoothWindow(1, 5), plot=plot)

    new_es = EventSeries.from_string(
        " |   A | B   |     |     |   A |     | \n"
        " |     |     |   A | B   |     |   A | \n"
        " |   A | B   |     |   A |   A |     | \n"
        " |     |   A |   A |   A |   A |     | \n"
        " | B   |   A | B   |     | B   |   A | \n"
        " | B   |   A | B   |     | B   |     |   ",
        model=es)
    new_es.warp_with_model(iterations=3)
    print(f"\nDone warping. {new_es.converged_str}")
    print("\n" + new_es.format_warped_series())

    es.compute_likelihoods(laplace_smoothing=0.1)
    llls = new_es.likelihood_with_model()
    for idx, lll in enumerate(llls):
        print(f'Likelihood[{idx}] = exp({lll:6.2f}) = {np.exp(lll):.5f}')
