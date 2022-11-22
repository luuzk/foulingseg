import json
import os
import random
from glob import glob
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras


### Environment configuration

def set_seeds(seed: int):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def allow_gpu_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)


def use_mixed_precision():
    keras.mixed_precision.set_global_policy('mixed_float16')
    policy = keras.mixed_precision.global_policy()
    print('Compute dtype: ', policy.compute_dtype)
    print('Variable dtype:', policy.variable_dtype)


def suppress_tensorflow_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### Path and parameter storage operations

def make_submission_path(results_dir: str, 
                         model_func, 
                         *args) -> str:
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    model_func = model_func.__name__ if callable(model_func) else str(model_func)

    submissions = sorted(glob(os.path.join(results_dir, "[0123456789]"*5 + "_*")))
    submissions = [sub for sub in submissions if os.path.isdir(sub)]
    last_submission = int(os.path.basename(submissions[-1]).split("_")[0]) if len(submissions) > 0 else 0
    
    path = os.path.join(results_dir, f"{last_submission + 1:05d}_{model_func}")
    for arg in args:
        path += "_" + str(arg).replace(" ", "_")
    os.mkdir(path)
    return path


def load_dict_if_path(dict_or_path: Union[str, dict],
                      save_dir: str) -> Tuple[dict, str]:
    if os.path.isfile(str(dict_or_path)):
        with open(dict_or_path) as f:
            loaded_dict = json.load(f)
        if not save_dir:
            save_dir = os.path.dirname(dict_or_path)
        return loaded_dict, save_dir
    else:
        assert isinstance(dict_or_path, dict)
        assert os.path.isdir(save_dir)
        return dict_or_path, save_dir


def save_params(save_path: str,
                *params,
                name: str = "params.txt"):
    save_path = os.path.abspath(save_path)
    assert os.path.isdir(save_path) or os.path.isdir(os.path.dirname(save_path))
    assert isinstance(name, str) and len(name) > 0
    assert len(params) > 0

    # Stringify if not serializable
    to_str = lambda x: ".".join((x.__module__, x.__name__)) \
        if callable(x) and hasattr(x, "__name__") else str(x)

    if len(params) == 1: params = params[0]
    if os.path.isdir(save_path): save_path = os.path.join(save_path, name)

    with open(save_path, "w") as f:
        json.dump(params, f, default=to_str, indent=4, sort_keys=True)


def cm_from_model(model_path: str):
    assert os.path.isfile(model_path)

    metrics = os.path.join(os.path.dirname(model_path), "metrics.json")
    with open(metrics) as f:
        metrics = json.load(f)

    return np.asarray(metrics["cm"])

### Visualization

def save_model_summary(model: keras.Model, 
                       save_path: str):
    assert(os.path.isdir(os.path.dirname(save_path)))
    # Generate summary
    lines = []
    model.summary(print_fn=lambda *x: lines.extend(x))
    model_summary = "\n".join(lines)

    # print(model_summary, flush=True)
    with open(save_path, "w") as f:
        f.writelines(model_summary)


def show_image_label(image: tf.Tensor, 
                     label: tf.Tensor, 
                     palette: list[int], 
                     grid: tuple[int, int] = (4, 4),
                     shift_range: tuple[int, int, int, int] = None) -> plt.Figure:
    '''Show tensors of images and their corresponding labels for visual inspection
    '''
    assert tf.rank(image) == 4 and tf.rank(label) == 4
    assert tf.shape(image)[-1] == 3 and tf.shape(label)[-1] == 1
    assert len(grid) == 2 and grid[1] % 2 == 0
    assert len(palette) % 3 == 0
    assert not shift_range or len(shift_range) == 4

    image, label = image.numpy(), label.numpy()
    if shift_range:
        a1, a2, b1, b2 = shift_range
        image = (image - a1) / (a2 - a1) # to [0, 1]
        image = image * (b2 - b1) + b1 # to desired range
    image = image.astype(np.uint8)
    label = label.astype(np.uint8)

    num_pairs = grid[0] * grid[1] // 2
    assert num_pairs <= image.shape[0]

    fig, axes = plt.subplots(*grid, figsize=(10,10))
    for idx, (ax1, ax2) in zip(range(num_pairs), axes.reshape((-1, 2))):
        ax1.set_axis_off()
        ax1.imshow(image[idx])

        ax2.set_axis_off()
        ax2.imshow(indexed_from_arr(label[idx,...,0], palette))
    fig.tight_layout(pad=0)
    return fig


def debug_output(datasets: list[tf.data.Dataset], 
                 names: list[str],
                 save_dir: str):
    assert len(datasets) > 0 and len(datasets) == len(names)

    palette = load_colormap()[1]

    for ds, ds_name in zip(datasets, names):
        _debug_im, _debug_lbl = next(iter(ds.unbatch().batch(32)))
        if isinstance(_debug_lbl, dict):
            _debug_lbl = _debug_lbl["mask"] 
        if isinstance(_debug_lbl, tuple):
            _debug_lbl = _debug_lbl[0] 
        fig = show_image_label(_debug_im[...,:3], _debug_lbl, palette=palette,
            grid=(8, 8)) #shift_range=(-1, 1, 0, 255))
        fig.savefig(os.path.join(save_dir, f"debug_{ds_name}_batch.png"))


def plot_confusion_matrix(cm: np.ndarray, 
                          labels: list[str], 
                          normalize: str = "true"):
    assert len(labels) == len(cm)
    
    cm = np.asarray(cm).astype(int)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1]
    assert np.min(cm) >= 0
    
    # Normalize
    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        else:
            raise ValueError("No valid specifier for 'normalize'")
        cm = np.nan_to_num(cm)
    
    # Plot
    plt.figure(figsize=(9.6,7.2)) # Multiple of default size for correct colorbar
    cmdisplay = ConfusionMatrixDisplay(cm, display_labels=labels)
    cmdisplay.plot(xticks_rotation="vertical", cmap="Blues", values_format=".2f", ax=plt.gca())


def save_confusion_matrix_plot(metrics: dict,
                               save_dir = None,
                               colormap_path = "data"):
    # Load data and classes from colormap
    metrics, save_dir = load_dict_if_path(metrics, save_dir)

    classes, _ = load_colormap(colormap_path)
    classes.pop("empty")
    classes = list(classes.keys())

    # Plot
    plot_confusion_matrix(metrics["cm"], classes)
    miou_key = next(k for k in metrics.keys() if k.endswith("mean_iou"))
    plt.title(f"mIoU {metrics[miou_key]:.3f}")

    # Save
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, pad_inches=0.05, bbox_inches="tight")


def plot_training_history(history: dict, 
                          fine_tuning_epoch: int = None):
    plt.figure(figsize=(8,8))
    epochs = list(range(1, len(history["loss"]) + 1))
    keys = [k for k in history.keys() if not k.startswith("val") and f"val_{k}" in history.keys()]

    # Plot history
    for k, c in zip(keys, ("b", "r", "c", "m", "g")):
        plt.plot(epochs, history[k], f"{c}-", label=f"training {k}")
        plt.plot(epochs, history[f"val_{k}"], f"{c}--", label=f"validation {k}")
    plt.ylim(0, 1.5)
    plt.xlim(1, len(epochs))
    if fine_tuning_epoch:
        plt.axvline(fine_tuning_epoch, label="Start Fine Tuning")
    plt.legend()

    # Plot best model
    best = np.argmin(history["val_loss"])
    plt.plot(epochs[best], history["val_loss"][best], "ko", fillstyle="none", ms=20)
    plt.title(f"Best val_loss of {history['val_loss'][best]:.4f} at epoch {epochs[best]}")


def save_training_history_plot(history: dict, 
                               save_dir: str = None, 
                               fine_tuning_epoch = None):
    history, save_dir = load_dict_if_path(history, save_dir)

    plot_training_history(history, fine_tuning_epoch=fine_tuning_epoch)

    save_path = os.path.join(save_dir, "history.png")
    plt.savefig(save_path, pad_inches=0.05, bbox_inches="tight")

### Segmentation mask methods

def load_colormap(path="data") -> Tuple[dict, list]:
    '''Loads (classes, palette) from 'path' if file or 'path/colormap.json' if path is a directory
    '''
    if os.path.isdir(path):
        path = os.path.join(path, "colormap.json")
    
    assert os.path.isfile(path), f"{path} not found"
    with open(path, "r") as f:
        classes, palette = json.load(f)
    return classes, palette


def indexed_from_arr(arr: np.ndarray, 
                     palette: list[int]) -> Image.Image:
    assert isinstance(arr, np.ndarray) and isinstance(palette, (list, tuple)) 
    assert np.min(arr) >= 0 and np.max(arr) <= 255
    assert np.max(arr) < len(palette) // 3, "array indices > number of colors"
    assert arr.ndim == 2

    im = Image.fromarray(arr.astype(np.uint8), mode="P")
    im.putpalette(palette)
    return im
