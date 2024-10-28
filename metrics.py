import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Callable, Any

import numpy as np
from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from functools import partial

import sklearn
import logging

logger = logging.getLogger()

# To transform predictions to one value per timepoint 
class ADMetric(ABC):
    """Base class for metric implementations that score anomaly scorings against ground truth binary labels. Every
    subclass must implement :func:`~timeeval.metrics.Metric.name`, :func:`~timeeval.metrics.Metric.score`, and
    :func:`~timeeval.metrics.Metric.supports_continuous_scorings`.
    """

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:  # type: ignore[no-untyped-def]
        y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)
        if np.unique(y_score).shape[0] == 1:
            warnings.warn("Cannot compute metric for a constant value in y_score, returning 0.0!")
            return 0.
        return self.score(y_true, y_score)

    def _validate_scores(self, y_true: np.ndarray, y_score: np.ndarray,
                         inf_is_1: bool = True,
                         neginf_is_0: bool = True,
                         nan_is_0: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.array(y_true).copy()
        y_score = np.array(y_score).copy()
        # check labels
        if self.supports_continuous_scorings() and y_true.dtype == np.float_ and y_score.dtype == np.int_:
            warnings.warn("Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
                          "y_true should be an integer array and y_score a float array!")
            return self._validate_scores(y_score, y_true)

        y_true: np.ndarray = column_or_1d(y_true)  # type: ignore
        assert_all_finite(y_true)

        # check scores
        y_score: np.ndarray = column_or_1d(y_score)  # type: ignore

        check_consistent_length([y_true, y_score])
        if not self.supports_continuous_scorings():
            if y_score.dtype not in [np.int_, np.bool_]:
                raise ValueError("When using Metrics other than AUC-metric that need discrete (0 or 1) scores (like "
                                 "Precision, Recall or F1-Score), the scores must be integers and should only contain "
                                 "the values {0, 1}. Please consider applying a threshold to the scores!")
        else:
            if y_score.dtype != np.float_:
                raise ValueError("When using continuous scoring metrics, the scores must be floats!")

        # substitute NaNs and Infs
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        penalize_mask = np.full_like(y_score, dtype=bool, fill_value=False)
        if inf_is_1:
            y_score[inf_mask] = 1
        else:
            penalize_mask = penalize_mask | inf_mask
        if neginf_is_0:
            y_score[neginf_mask] = 0
        else:
            penalize_mask = penalize_mask | neginf_mask
        if nan_is_0:
            y_score[nan_mask] = 0.
        else:
            penalize_mask = penalize_mask | nan_mask
        y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

        assert_all_finite(y_score)
        return y_true, y_score

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this metric."""
        ...

    @abstractmethod
    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Implementation of the metric's scoring function.

        Please use :func:`~timeeval.metrics.Metric.__call__` instead of calling this function directly!

        Examples
        --------

        Instantiate a metric and call it using the ``__call__`` method:

        >>> import numpy as np
        >>> from timeeval.metrics import RocAUC
        >>> metric = RocAUC(plot=False)
        >>> metric(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.5

        """
        ...

    @abstractmethod
    def supports_continuous_scorings(self) -> bool:
        """Whether this metric accepts continuous anomaly scorings as input (``True``) or binary classification
        labels (``False``)."""
        ...
        

class AucMetric(ADMetric, ABC):
    """Base class for area-under-curve-based metrics.

    All AUC-Metrics support continuous scorings, calculate the area under a curve function, and allow plotting this
    curve function. See the subclasses' documentation for a detailed explanation of the corresponding curve and metric.
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        self._plot = plot
        self._plot_store = plot_store

    def _auc(self,
             y_true: np.ndarray,
             y_score: Iterable[float],
             curve_function: Callable[[np.ndarray, np.ndarray], Any]) -> float:
        x, y, thresholds = curve_function(y_true, np.array(y_score))
        if "precision_recall" in curve_function.__name__:
            # swap x and y
            x, y = y, x
        area: float = auc(x, y)
        if self._plot:
            import matplotlib.pyplot as plt

            name = curve_function.__name__
            plt.plot(x, y, label=name, drawstyle="steps-post")
            # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.title(f"{name} | area = {area:.4f}")
            if self._plot_store:
                plt.savefig(f"fig-{name}.pdf")
            plt.show()
        return area

    def supports_continuous_scorings(self) -> bool:
        return True


# Area Under the curve!
class RocAUC(AucMetric):
    """Computes the area under the receiver operating characteristic curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".

    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, roc_curve)

    @property
    def name(self) -> str:
        return "ROC_AUC"


class PrAUC(AucMetric):
    """Computes the area under the precision recall curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, precision_recall_curve)

    @property
    def name(self) -> str:
        return "PR_AUC"

def get_metric_scores(metrics, Y_test, scores, scores_val):
    results = []

    tpv_3_callback = partial(tpv_F1_3_std, scores_val)
    tpv_2_callback = partial(tpv_F1_2_std, scores_val)
    tpv_1_callback = partial(tpv_F1_1_std, scores_val)

    metric_func = {
        'best f1 (F1)': best_f1_F1,
        'trivial val 3 Std (F1)' : tpv_3_callback,
        'trivial val 2 Std (F1)' : tpv_2_callback,
        'trivial val 1 Std (F1)' : tpv_1_callback,
        '0.5 threshold (F1)': threshold_0_5_F1,
        'roc auc': roc_auc,
        'pr auc': pr_auc,
    }
    for metric in metrics:
        results.append(metric_func[metric](Y_test, scores))
    return results

def best_f1_F1(Y_test, scores):
    t = best_f_score(scores, Y_test)
    logger.info("Best F1 threshold : %s", t)
    predictions = scores >= t
    return f1_score(Y_test, predictions)

def best_f1_ACC(Y_test, scores):
    t = best_f_score(scores, Y_test)
    predictions = scores >= t
    return (Y_test == predictions).float().mean().item()

def threshold_0_5_F1(Y_test, scores):
    t = 0.5
    predictions = scores >= t
    return f1_score(Y_test, predictions)

def tpv_F1_3_std(scores_val, Y_test, scores):
    t = trivial_percentile_val(scores_val)
    logger.info("Trivial percentile value 3std : %s", t)
    predictions = scores >= t
    return f1_score(Y_test, predictions)

def tpv_F1_2_std(scores_val, Y_test, scores):
    t = trivial_percentile_val(scores_val, 95)
    logger.info("Trivial percentile value 2std : %s", t)
    predictions = scores >= t
    return f1_score(Y_test, predictions)

def tpv_F1_1_std(scores_val, Y_test, scores):
    t = trivial_percentile_val(scores_val, 68)
    logger.info("Trivial percentile value 1std : %s", t)
    predictions = scores >= t
    return f1_score(Y_test, predictions)

def roc_auc(Y_test, scores):
    auroc, _ = calculate_auc(Y_test, scores)
    return auroc

def pr_auc(Y_test, scores):
    _, aupr = calculate_auc(Y_test, scores)
    return aupr

def top_k(scores, anomaly_ratio):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    """
    return np.percentile(scores, 100 - anomaly_ratio)

def get_f_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def trivial_percentile_val(scores_val, p_val=99.7):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param p_val: 95 is two stds, 99.7 is 3 stds 
    """
    return np.percentile(scores_val, p_val)

def calculate_auc(test_labels, test_scores):
    # measure prediction performance before threshold calculation
    auroc = RocAUC().score(test_labels, test_scores) 
    aupr = PrAUC().score(test_labels, test_scores)
    return auroc, aupr

def best_f_score(scores, targets):
    """
    :param scores: list or np.array or tensor, test anomaly scores
    :param targets: list or np.array or tensor, test target labels
    :return: max threshold
    """
    prec, rec, thresholds = precision_recall_curve(targets, scores)
    
    fscores = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
    #Remove nans
    fscores = np.nan_to_num(fscores)
    opt_num = np.squeeze(np.argmax(fscores))
    opt_thres = thresholds[opt_num]          
    return opt_thres

def f1_score(Y, Y_pred):
    score = sklearn.metrics.f1_score(Y, Y_pred)
    if score == 0:
        logger.warning("F1 SCORE ZERO, Y = %s, Y_pred = %s", Y, Y_pred)
    return score

def roc_auc_score(Y, Y_pred):
    return sklearn.metrics.roc_auc_score(Y, Y_pred)
