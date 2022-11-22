import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.python.ops import array_ops, math_ops


class MeanIoU(keras.metrics.MeanIoU):
    '''Wrapper for keras.metrics.MeanIoU handling sparse and weighted labels
    '''
    def __init__(self, num_classes, average="macro", name="mean_iou", dtype=None):
        super(MeanIoU, self).__init__(num_classes, name=name, dtype=dtype)
        assert average in {"micro", "macro", "weighted", "macro_per_class"}
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast input and flatten
        y_true = K.reshape(K.cast(y_true, self._dtype), (-1,))
        y_pred = K.reshape(K.argmax(y_pred, axis=-1), (-1,))

        # Remove empty class (label=0) indices from accumulation
        mask = K.greater(y_true, 0.)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Shift other indices towards zero
        y_true = K.maximum(y_true - 1., 0.)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), self._dtype)
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), self._dtype)
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        if self.average == "micro":
            true_positives = math_ops.reduce_sum(true_positives)
            denominator = math_ops.reduce_sum(denominator)

        iou = math_ops.div_no_nan(true_positives, denominator)

        if self.average == "macro_per_class":
            return iou

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(
            math_ops.cast(math_ops.not_equal(denominator, 0), self._dtype))

        # Use the number of true instances per label as weight
        if self.average == "weighted":
            iou = math_ops.mul(iou, sum_over_col)
            sum_weights = math_ops.reduce_sum(sum_over_col)
            return math_ops.div_no_nan(
                math_ops.reduce_sum(iou, name='mean_iou'), sum_weights)

        return math_ops.div_no_nan(
            math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)


def average_metrics(metrics: list):
    assert isinstance(metrics, (list, tuple)) and len(metrics) > 0
    assert all(isinstance(m, dict) for m in metrics)
    assert all(m.keys() == metrics[0].keys() for m in metrics)

    means = {}
    stds = {}
    for key in metrics[0].keys():
        values = np.asarray([m[key] for m in metrics])

        means[key] = np.mean(values, axis=0).astype(values.dtype).tolist()
        stds[key] = np.std(values, axis=0).tolist()

    return means, stds
