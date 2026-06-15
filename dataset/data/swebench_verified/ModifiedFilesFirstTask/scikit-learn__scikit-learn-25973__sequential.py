"""
Sequential feature selection
"""
from numbers import Integral, Real

import numpy as np

import warnings

from ._base import SelectorMixin
from ..base import BaseEstimator, MetaEstimatorMixin, clone
from ..utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..utils._param_validation import RealNotInt
from ..utils._tags import _safe_tags
from ..utils.validation import check_is_fitted
from ..model_selection import cross_val_score
from ..metrics import get_scorer_names


class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Transformer that performs Sequential Feature Selection.

    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator. In
    the case of unsupervised learning, this Sequential Feature Selector
    looks only at the features (X), not the desired outputs (y).

    Read more in the :ref:`User Guide <sequential_feature_selection>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.

    n_features_to_select : "auto", int or float, default='warn'
        If `"auto"`, the behaviour depends on the `tol` parameter:

        - if `tol` is not `None`, then features are selected until the score
          improvement does not exceed `tol`.
        - otherwise, half of the features are selected.

        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

        .. versionadded:: 1.1
           The option `"auto"` was added in version 1.1.

        .. deprecated:: 1.1
           The default changed from `None` to `"warn"` in 1.1 and will become
           `"auto"` in 1.3. `None` and `'warn'` will be removed in 1.3.
           To keep the same behaviour as `None`, set
           `n_features_to_select="auto" and `tol=None`.

    tol : float, default=None
        If the score is not incremented by at least `tol` between two
        consecutive feature additions or removals, stop adding or removing.

        `tol` can be negative when removing features using `direction="backward"`.
        It can be useful to reduce the number of features at the cost of a small
        decrease in the score.

        `tol` is enabled only when `n_features_to_select` is `"auto"`.

        .. versionadded:: 1.1

    direction : {'forward', 'backward'}, default='forward'
        Whether to perform forward selection or backward selection.

    scoring : str or callable, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        NOTE that when using a custom scorer, it should return a single
        value.

        If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. When evaluating a new feature to
        add or remove, the cross-validation procedure is parallel over the
        folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_features_to_select_ : int
        The number of features that were selected.

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    See Also
    --------
    GenericUnivariateSelect : Univariate feature selector with configurable
        strategy.
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination based on importance weights, with
        automatic selection of the number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.

    Examples
    --------
    >>> from sklearn.feature_selection import SequentialFeatureSelector
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
    >>> sfs.fit(X, y)
    SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                              n_features_to_select=3)
    >>> sfs.get_support()
    array([ True, False,  True,  True])
    >>> sfs.transform(X).shape
    (150, 3)
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_features_to_select": [
            StrOptions({"auto", "warn"}, deprecated={"warn"}),
            Interval(RealNotInt, 0, 1, closed="right"),
            Interval(Integral, 0, None, closed="neither"),
            Hidden(None),
        ],
        "tol": [None, Interval(Real, None, None, closed="neither")],
        "direction": [StrOptions({"forward", "backward"})],
        "scoring": [None, StrOptions(set(get_scorer_names())), callable],
        "cv": ["cv_object"],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select="warn",
        tol=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
    ):

        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.tol = tol
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Implement your code here
        """
        return None

    def _get_best_new_feature_score(self, estimator, X, y, current_mask):
        """
        Implement your code here
        """
        return None

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _more_tags(self):
        return {
            "allow_nan": _safe_tags(self.estimator, key="allow_nan"),
        }
