import tensorflow.keras.backend as K # pylint: disable=no-name-in-module
from tensorflow import keras


class SparseCategoricalCrossentropyDiceLoss(keras.losses.Loss):
    '''Implementation based on https://arxiv.org/pdf/1809.10486.pdf
    '''
    def __init__(self, dice_weight=0.5, **kwargs):
        super().__init__(**kwargs)

        assert 0 <= dice_weight <= 1
        self.dice_weight = dice_weight

    def call(self, y_true, y_pred):
        # Remove last (sparse) dim of size 1 in (B, W, H, 1)
        y_true = K.squeeze(y_true, axis=-1)

        # Only backprop non-empty label pixels
        mask = K.cast_to_floatx(y_true > 0) # (B, W, H)

        # Shift class indices, because no empty class (label=0) exists in y_pred
        y_true = K.maximum(y_true - 1, 0)
        # Labels to one-hot tensor
        y_true = K.one_hot(y_true, K.shape(y_pred)[-1]) # (B, W, H, K)

        # Categorical crossentropy
        ce = keras.losses.categorical_crossentropy(y_true, y_pred) # (B, W, H)
        ce = K.mean(ce * mask, axis=(1, 2)) # (B,)

        # Multi-class Dice loss
        mask = K.expand_dims(mask) # (B, W, H, 1)
        intersection = K.sum(mask * (y_true * y_pred), axis=(1, 2)) # (B, K)
        denom = K.sum(mask * (y_true + y_pred), axis=(1, 2)) # (B, K)

        # Compute the mean only over classes that appear in the label or
        # the prediction (union)
        num_valid_classes = K.sum(K.cast_to_floatx(K.sum(denom, axis=0) > 0.))
        dice = K.sum(intersection / (denom + K.epsilon()), axis=-1) / num_valid_classes # (B,)
        dice = 1 - 2 * dice

        return (1. - self.dice_weight) * ce + self.dice_weight * dice


class SparseCategoricalCrossentropy(keras.losses.SparseCategoricalCrossentropy):
    def call(self, y_true, y_pred):
        # Remove last (sparse) dim of size 1 in (B, W, H, 1)
        y_true = K.squeeze(y_true, axis=-1)

        # Only backprop non-empty label pixels
        mask = K.cast_to_floatx(y_true > 0) # (B, W, H)

        # Shift class indices, because no empty class (label=0) exists in y_pred
        y_true = K.maximum(y_true - 1, 0)

        # Sparse Categorical crossentropy
        ce = super().call(y_true, y_pred)
        return mask * ce
