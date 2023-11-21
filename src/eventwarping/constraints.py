import numpy as np
from abc import ABC, abstractmethod


class ConstraintsBaseClass(ABC):
    def __init__(self, es=None):
        self._es = es

    @property
    def es(self):
        if self._es is None:
            raise Exception(f"EventSeries ('es') need to be set in a constraint before it can be used.")
        return self._es

    @es.setter
    def es(self, value):
        self._es = value

    @abstractmethod
    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        raise NotImplementedError()

    # @abstractmethod
    # def calculate_constraint_matrix(self):
    #     """Constraint matrix has same form as cumulative cost matrix. Cell is
    #     True if corresponding realignment violates constraint and False otherwise.
    #
    #     ==DEPRECATED?==
    #     """
    #     raise NotImplementedError()


class MaxMergeSymbolConstraints(ConstraintsBaseClass):
    def __init__(self, k=1, es=None):
        self.k = k
        super().__init__(es=es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        return np.all(merged_cnts_s <= self.k)


class MaxMergeEventConstraints(ConstraintsBaseClass):
    def __init__(self, k=1, es=None):
        self.k = k
        super().__init__(es=es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        return np.all(merged_cnts_e <= self.k)


class MaxMergeSymbolSetConstraints(ConstraintsBaseClass):
    def __init__(self, k=1, symbols=None, es=None):
        self.k = k
        self.symbols = symbols
        super().__init__(es=es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        return np.all(merged_cnts_s[self.symbols] <= self.k)


# class MaxMergeConstraints(ConstraintsBaseClass):
#     def __init__(self, p=3, per_symbol=True, es=None):
#         super().__init__(es=es)
#         self.p = p
#         self.constraint_matrix = None
#         self.per_symbol = per_symbol
#
#     def calculate_constraint_matrix(self):
#         self.constraint_matrix = np.zeros([self.es.nb_series, self.es.nb_events, 3], dtype=bool)
#         self.add_no_more_than_p_symbols_together_constraint()
#         return self.constraint_matrix
#
#     def add_no_more_than_p_symbols_together_constraint(self):
#         for j, series in enumerate(self.es.warped_series):
#             item_count = self.es.warped_series[j, :-1, :] + self.es.warped_series[j, 1:, :]
#
#             if self.per_symbol:
#                 condition = np.sum((item_count > self.p), 1).astype(bool)
#             else:
#                 # TODO: still allows to do |A| |B| to become | |AB| | if p=1
#                 condition = (np.sum(item_count, axis=1) > self.p).astype(bool)
#             self.constraint_matrix[j, 1:, 0][condition] = True
#             self.constraint_matrix[j, :-1, 2][condition] = True
#
#
# class ApneaConstraints(ConstraintsBaseClass):
#
#     def calculate_constraint_matrix(self):
#         self.constraint_matrix = np.zeros([self.es.nb_series, self.es.nb_events, 3], dtype=bool)
#         self.add_start_stop_constraint([1, 3], [2, 4])
#         self.add_no_more_than_p_breaths_together_constraint(3)
#         return self.constraint_matrix
#
#     def add_start_stop_constraint(self, start, stop):
#         # True if realignment puts a start and stop in the same itemset
#         if isinstance(start, int) and isinstance(stop, int):
#             for j, series in enumerate(self.es.warped_series):
#                 start_end_same_itemset = ((series[:-1, start] > 0) & (series[1:, stop] > 0))
#                 self.constraint_matrix[j, 1:, 0][start_end_same_itemset] = True
#                 self.constraint_matrix[j, :-1, 2][start_end_same_itemset] = True
#         elif isinstance(start, list) and isinstance(stop, list) and (len(start) == len(stop)):
#             for j, series in enumerate(self.es.warped_series):
#                 for i in range(len(start)):
#                     start_end_same_itemset = ((series[:-1, start[i]] > 0) & (series[1:, stop[i]] > 0))
#                     self.constraint_matrix[j, 1:, 0][start_end_same_itemset] = True
#                     self.constraint_matrix[j, :-1, 2][start_end_same_itemset] = True
#         else:
#             raise ValueError
#
#     def add_no_more_than_p_breaths_together_constraint(self, p=3):
#         # True if more than p breaths together in an itemset
#         for j, series in enumerate(self.es.warped_series):
#             item_count = self.es.warped_series[j, :-1, :] + self.es.warped_series[j, 1:, :]
#
#             item_count[:, 5] = np.sum((item_count[:, 5:12]), 1)
#             item_count[:, 12] = np.sum((item_count[:, 12:19]), 1)
#             item_count[:, 19] = np.sum((item_count[:, 19:26]), 1)
#             item_count[:, 26] = np.sum((item_count[:, 26:33]), 1)
#
#             condition = np.sum((item_count > p), 1).astype(bool)
#             self.constraint_matrix[j, 1:, 0][condition] = True
#             self.constraint_matrix[j, :-1, 2][condition] = True

