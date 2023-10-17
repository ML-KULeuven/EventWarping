import pytest
import numpy as np
from pathlib import Path

from eventwarping.eventseries import EventSeries


def test_example1():
    print("")
    fn = Path(__file__).parent / "rsrc" / "example1.txt"
    es = EventSeries.from_file(fn)
    # es.print_series()
    # print(es.series.shape)

    print(f"Windowed counts: {es.windowed_counts}")
    wc_sol = np.array([5, 2, 3, 4, 3, 2])
    np.testing.assert_array_equal(es.windowed_counts, wc_sol)

    print(f"Warping directions: {es.warping_directions}")
    # wd_sol = np.array([-3., -1., 1., 0., -1., -1.])
    # np.testing.assert_array_equal(es.warping_directions, wd_sol)

