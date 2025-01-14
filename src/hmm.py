import warnings

import numpy as np
from hmmlearn.base import _BaseHMM
from hmmlearn.utils import normalize
from sklearn.utils import check_random_state


class NewCategoricalHMM(_BaseHMM):
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

    # score_samples, score, decode, predict, predict_proba, sample, fit = map(
    #     _multinomialhmm_fix_docstring_shape, [
    #         _BaseHMM.score_samples,
    #         _BaseHMM.score,
    #         _BaseHMM.decode,
    #         _BaseHMM.predict,
    #         _BaseHMM.predict_proba,
    #         _BaseHMM.sample,
    #         _BaseHMM.fit,
    #     ])

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _init(self, X, lengths=None):
        self._check_and_set_n_features(X)
        super()._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
            # self.emissionprob_ = self.random_state \
            #     .rand(self.n_components, self.n_features)
            # normalize(self.emissionprob_, axis=1)
            self.emissionprob_ = np.random.random((self.n_components, self.n_features))

    def _check(self):
        super()._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features

    def _compute_log_likelihood(self, X):
        # # remove empty rows
        # X_temp = X[X.any(axis=1)]
        # X = np.pad(X_temp, ((0,len(X) - len(X_temp)), (0,0)))

        # self.emissionprob_[-1] = 10**(-10)    # last component corresponds to empty timepoint
        self.emissionprob_ = np.clip(self.emissionprob_, 10**(-10), 1 - 10**(-10)) # avoid to small/large probabilities

        ll = X.dot(np.log(self.emissionprob_).T) + (1-X).dot(np.log(1-self.emissionprob_).T)

        if np.any(np.isnan(ll)):
            print(1)

        return X.dot(np.log(self.emissionprob_).T) + (1-X).dot(np.log(1-self.emissionprob_).T)

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            stats['obs'] += posteriors.T.dot(X)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'e' in self.params:
            self.emissionprob_ = (
                    stats['obs'] / stats['obs'].sum(axis=1, keepdims=True))

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a Multinomial distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if hasattr(self, "n_features"):
            if self.n_features != X.shape[1]:
                raise ValueError(
                    "N_features not equal to size of data")
        self.n_features = X.shape[1]
