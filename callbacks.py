import os
from glob import glob
from shutil import copy

import numpy as np
from PIL import Image
from tensorflow import keras

from dataset import load_dataset_np
from utils import indexed_from_arr, load_colormap


class EarlyStopping(keras.callbacks.EarlyStopping):
    '''Fixes issue which prevents always restoring best weights
    See https://github.com/keras-team/keras/issues/12511
    '''
    __doc__ = keras.callbacks.EarlyStopping.__doc__

    def on_train_end(self, logs=None):
        super().on_train_end(logs=logs)
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)

    # pylint: disable=attribute-defined-outside-init
    def on_train_begin(self, logs):
        super().on_train_begin(logs=logs)
        if self.restore_best_weights:
            self.best_weights = self.model.get_weights()


class SampleVisualization(keras.callbacks.Callback):
    '''Visualize the training progress by prediction of a few samples at the end
    of every epoch.
    '''
    def __init__(self,
                 save_dir,
                 samples_dir,
                 palette = load_colormap()[1],
                 preproc_func=lambda x: x,
                 postproc_func=lambda x: np.argmax(x, axis=-1) + 1, # shift indices
                 interval="best",
                 copy_masks=True,
                 make_gif=False,
                 downsample=None):
        super(SampleVisualization, self).__init__()

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

        assert os.path.isdir(samples_dir)
        sample_paths = glob(os.path.join(samples_dir, "*.JPG"))
        self.sample_names = [os.path.splitext(os.path.basename(p))[0] for p in sample_paths]

        # Copy test samples & masks to saving dir
        for path in sample_paths:
            copy(path, save_dir)
            if copy_masks:
                copy(os.path.splitext(path)[0] + ".png", save_dir)

        # Load test samples
        self.test_samples = preproc_func(load_dataset_np(sample_paths, downsample=downsample))

        assert isinstance(interval, int) or interval == "best"
        self.interval = interval
        self.best_loss = np.inf

        self.palette = palette
        self.postproc_func = postproc_func
        self.make_gif = make_gif

    def _save_test_masks(self, suffix):
        # Predict masks
        masks = self.model.predict_on_batch(self.test_samples)
        masks = self.postproc_func(masks)
        assert masks.ndim == 3

        # Save masks
        for mask, name in zip(masks, self.sample_names):
            save_path = f"{name}{suffix}.png"
            save_path = os.path.join(self.save_dir, save_path)

            im = indexed_from_arr(mask, self.palette)
            im.save(save_path)

    # pylint: disable=unused-argument
    def on_train_begin(self, logs=None):
        self._save_test_masks("_000")

    # pylint: disable=unused-argument
    def on_train_end(self, logs=None):
        self._save_test_masks("_final")

        if self.make_gif:
            for name in self.sample_names:
                # Find PNG masks for GIF
                imgs = glob(os.path.join(self.save_dir, f"{name}_*.png"))
                img, *imgs = [Image.open(im) for im in sorted(imgs)]
                # Save GIF
                img.save(os.path.join(self.save_dir, f"{name}.gif"), 
                    append_images=imgs, save_all=True, loop=0, duration=500)

    def on_epoch_end(self, epoch, logs=None):
        suffix = f"_{epoch + 1:03d}"

        if self.interval == "best":
            if logs["val_loss"] <= self.best_loss:
                self.best_loss = logs["val_loss"]
                self._save_test_masks(suffix)
        else:
            if not (epoch + 1) % self.interval:
                self._save_test_masks(suffix)
