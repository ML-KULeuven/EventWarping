import numpy as np
from abc import ABC, abstractmethod


class ConstraintBaseClass(ABC):
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


class MaxMergeSymbolConstraint(ConstraintBaseClass):
    """Allow merging events if no symbol is merged more than k times."""
    def __init__(self, k=1, es=None):
        self.k = k
        super().__init__(es=es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        if a is None or b is None:
            return True
        return np.all(merged_cnts_s <= self.k)


class MaxMergeEventConstraint(ConstraintBaseClass):
    """Allow merging up to k events."""
    def __init__(self, k=1, es=None):
        self.k = k
        super().__init__(es=es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        if a is None or b is None:
            return True
        return np.all(merged_cnts_e <= self.k)


class MaxMergeEventIfSameConstraint(ConstraintBaseClass):
    """Allow merging up to k events if the events are identical."""
    def __init__(self, k=1, es=None):
        self.k = k
        super().__init__(es=es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        if a is None or b is None:
            return True
        if a is not None and np.any(a) and np.any(b) and not np.array_equal(np.sign(a), np.sign(b)):
            return False
        return np.all(merged_cnts_e <= self.k)


class MaxMergeSymbolSetConstraint(ConstraintBaseClass):
    """Allow merging events if none of the given symbols are merged more than k times."""
    def __init__(self, k=1, symbols=None, es=None):
        self.k = k
        super().__init__(es=es)
        self.symbols = self._translate_symbols(symbols, self._es)

    @classmethod
    def _translate_symbols(cls, symbols, es):
        if es is None:
            return symbols
        if es.symbol2int is None:
            return symbols
        return [es.symbol2int.get(s, s) for s in symbols]

    @ConstraintBaseClass.es.setter
    def es(self, value):
        self._es = value
        self.symbols = self._translate_symbols(self.symbols, self._es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        if a is None or b is None:
            return True
        return np.all(merged_cnts_s[self.symbols] <= self.k)


class NoXorMergeSymbolSetConstraint(ConstraintBaseClass):
    """Allow merging events if none of the given symbols are merged with each other,
    except if it is the same symbol.

    Thus: |A|A| to |A|| is allowed,
    But: |A|B| to |AB|| is not if self.symbols is [A,B]
    """
    def __init__(self, symbols=None, es=None):
        super().__init__(es=es)
        self.symbols = self._translate_symbols(symbols, self._es)

    @classmethod
    def _translate_symbols(cls, symbols, es):
        if es is None:
            return symbols
        if es.symbol2int is None:
            return symbols
        return [es.symbol2int.get(s, s) for s in symbols]

    @ConstraintBaseClass.es.setter
    def es(self, value):
        self._es = value
        self.symbols = self._translate_symbols(self.symbols, self._es)

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        if a is None or b is None:
            return True
        if np.sum(np.sign(merged_cnts_s[self.symbols])) > 1:
            return False
        return True


class NoMergeTooDistantSymbolSetConstraint(ConstraintBaseClass):
    """Allow merging events if distance between sets is not larger than k."""
    def __init__(self, distance_function=None, k=1, es=None):
        self.distance_function = distance_function
        self.k = k
        super().__init__(es=es)

    def _translate_to_symbols(self, array):
        if self.es is None or len(self.es.symbol2int) == 0:
            return set(np.argwhere(array).flatten())
        int2symbol = {v:k for k,v in self.es.symbol2int.items()}
        return {int2symbol.get(i, i) for i in np.argwhere(array).flatten()}

    def allow_merge(self, merged_cnts_s, merged_cnts_e, a, b):
        a = self._translate_to_symbols(a)
        b = self._translate_to_symbols(b)
        return self.distance_function(a, b) <= self.k
