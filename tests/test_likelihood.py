import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os

from eventwarping.eventseries import EventSeries
from eventwarping.constraints import *
from eventwarping.window import LinearScalingWindow, StaticWindow, MultipleWindow


# If environment variable TESTDIR is set, save figures to this
# directory, otherwise to the test directory
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))


def test_likelihood7():
    """Warping leads to

         |     | B   |     |   A |   A |     |
         |   A | B   |     |     |   A |   A |
         |   A | B   |     |     |   A |     |

    """
    # window_type = "static"
    window_type = "linear"
    # window_type = "multiple"
    fn = Path(__file__).parent / "rsrc" / "example7.txt"
    window = 5
    if window_type == "linear":
        window = LinearScalingWindow(5)
    elif window_type == "static":
        window = 5
    elif window_type == "multiple":
        window = MultipleWindow([5, 1], [5, 5], delay="convergence")
    es = EventSeries.from_file(fn, window=window, constraints=[MaxMergeEventConstraint(1)])
    # print('Original:\n{es.format_warped_series()}')

    nb_iterations = 4
    plot = {'filename': str(directory / 'gradients_{iteration}.png'), 'symbol': {0, 1}} # , 'seriesidx': 1}
    # plot = None
    es.warp(iterations=nb_iterations, plot=plot)
    print(f"\nDone warping model. {es.converged_str}")

    if window_type == "static":
        # Do one additional iteration to update the gradients with a different window
        plot = {'filename': str(directory / f'gradients_{nb_iterations}.png'), 'symbol': {0, 1}}
        es.update_gradients_without_warping(window=StaticWindow(1, 5), plot=plot)

    new_es = EventSeries.from_string(
        " |   A | B   |     |     |   A |     | \n"
        " |     |     |   A | B   |     |   A | \n"
        " |   A | B   |     |   A |   A |     | \n"
        " |     |   A |   A |   A |   A |     | \n"
        " | B   |   A | B   |     | B   |   A | \n"
        " | B   |   A | B   |     | B   |     |   ",
        model=es)
    new_es.warp_with_model(iterations=3)
    print(f"\nDone warping new data. {new_es.converged_str}")
    print("\n" + new_es.format_warped_series())

    es.compute_likelihoods(laplace_smoothing=0.1)
    llls = new_es.likelihood_with_model()
    for idx, lll in enumerate(llls):
        print(f'Likelihood[{idx}] = exp({lll:8.4f}) = {np.exp(lll):.5f}')

    llls_sol = [-1.6764, -1.6764, -2.3230, -6.4036, -8.5443, -11.9783]
    for lll, lll_sol in zip(llls, llls_sol):
        assert lll == pytest.approx(lll_sol, 4)
