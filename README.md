# Event Warping

Align series of sets of events.

When using this tool, please cite:

> TODO

## Installation

TODO

## Usage

    >>> from eventwarping.eventseries import EventSeries
    >>> setlists = [[{'A', 'B'}, {}, {'A'}], [{'A'}, {'B'}, {'A'}]]
    >>> es = EventSeries.from_setlist(setlists, window=3)
    >>> es.compute_warped_series()
    >>> print(es.format_warped_series())
     B A |     |   A |
     B A |     |   A |


## License

Copyright 2023, KU Leuven, MIT

## Contact

Wannes Meert, Dries Van der Plas  
DTAI, Dept CS, KU Leuven

