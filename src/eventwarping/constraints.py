import numpy as np
from abc import ABC, abstractmethod


class ConstraintsBaseClass(ABC):
    def __init__(self, es):
        self.es = es
        self.nb_series, self.nb_events, self.nb_symbols = es.warped_series.shape

    @abstractmethod
    def calculate_constraint_matrix(self):
        """Constraint matrix has same form as cumulative cost matrix. Cell is
        True if corresponding realignment violates constraint and False otherwise."""
        raise NotImplementedError()


class ApneaConstraints(ConstraintsBaseClass):

    def calculate_constraint_matrix(self):
        self.constraint_matrix = np.zeros([self.nb_series, self.nb_events, 3], dtype=bool)
        self.add_start_stop_constraint([1, 3], [2, 4])
        self.add_no_more_than_p_breaths_together_constraint(3)
        return self.constraint_matrix

    def add_start_stop_constraint(self, start, stop):
        # True if realignment puts a start and stop in the same itemset
        if isinstance(start, int) and isinstance(stop, int):
            for j, series in enumerate(self.es.warped_series):
                start_end_same_itemset = ((series[:-1, start] > 0) & (series[1:, stop] > 0))
                self.constraint_matrix[j, 1:, 0][start_end_same_itemset] = True
                self.constraint_matrix[j, :-1, 2][start_end_same_itemset] = True
        elif isinstance(start, list) and isinstance(stop, list) and (len(start) == len(stop)):
            for j, series in enumerate(self.es.warped_series):
                for i in range(len(start)):
                    start_end_same_itemset = ((series[:-1, start[i]] > 0) & (series[1:, stop[i]] > 0))
                    self.constraint_matrix[j, 1:, 0][start_end_same_itemset] = True
                    self.constraint_matrix[j, :-1, 2][start_end_same_itemset] = True
        else:
            raise ValueError

    def add_no_more_than_p_breaths_together_constraint(self, p=3):
        # True if more than p breaths together in an itemset
        for j, series in enumerate(self.es.warped_series):
            item_count = self.es.warped_series[j, :-1, :] + self.es.warped_series[j, 1:, :]

            item_count[:, 5] = np.sum((item_count[:, 5:12]), 1)
            item_count[:, 12] = np.sum((item_count[:, 12:19]), 1)
            item_count[:, 19] = np.sum((item_count[:, 19:26]), 1)
            item_count[:, 26] = np.sum((item_count[:, 26:33]), 1)

            condition = np.sum((item_count > p), 1).astype(bool)
            self.constraint_matrix[j, 1:, 0][condition] = True
            self.constraint_matrix[j, :-1, 2][condition] = True

