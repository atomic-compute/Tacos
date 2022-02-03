import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.python.framework import ops


def l2_regularization_loss(weights, scale, blacklist):
    def is_black(name):
        return any([black in name for black in blacklist])

    target_weights = [tf.nn.l2_loss(w) for w in weights if not is_black(w.name)]
    l2_loss = sum(target_weights) * scale
    tf.losses.add_loss(l2_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
    return l2_loss


class ComputeWeightedLoss(losses.Loss):
    def __init__(
        self,
        weights=1.0,
        scope=None,
        loss_collection=None,
        reduction=None,
    ) -> None:
        self.weights = weights
        self.scope = scope
        self.loss_collection = loss_collection
        self.reduction = reduction

    def call(self, y_true, y_pred):
        # TODO: FixMe -> Write operation custom
        """
        Computes the weighted loss.

        Args:
            losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
            weights: Optional `Tensor` whose rank is either 0, or the same rank as
            `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `losses` dimension).
            scope: the scope for the operations performed in computing the loss.
            loss_collection: the loss will be added to these collections.
            reduction: Type of reduction to apply to loss.

        Returns:
            Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
            `NONE`, this has the same shape as `losses`; otherwise, it is scalar.

        Raises:
            ValueError: If `weights` is `None` or the shape is not compatible with
            `losses`, or if the number of dimensions (rank) of either `losses` or
            `weights` is missing.

        Note:
            When calculating the gradient of a weighted loss contributions from
            both `losses` and `weights` are considered. If your `weights` depend
            on some model parameters but you do not want this to affect the loss
            gradient, you need to apply `tf.stop_gradient` to `weights` before
            passing them to `compute_weighted_loss`.
        """
        # Reduction.validate(reduction)

        # losses = tf.convert_to_tensor(losses)
        # input_dtype = losses.dtype
        # losses = tf.cast(losses, dtype=tf.float32)
        # weights = tf.cast(weights, dtype=tf.float32)
        # weighted_losses = tf.multiply(losses, weights)
        # if reduction == tf.losses.Reduction.NONE:
        #     loss = weighted_losses
        # else:
        #     loss = tf.reduce_sum(weighted_losses)
        #     if reduction == tf.losses.Reduction.SUM:
        #         loss = _safe_mean(loss, tf.reduce_sum(array_ops.ones_like(losses) * weights))
        #     elif (reduction == Reduction.SUM_BY_NONZERO_WEIGHTS or
        #             reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS):
        #         loss = _safe_mean(loss, _num_present(losses, weights))
        #     elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        #         loss = _safe_mean(loss, _num_elements(losses))

        # # Convert the result back to the input type.
        # loss = tf.cast(loss, input_dtype)
        # util.add_loss(loss, loss_collection)
        # return loss
