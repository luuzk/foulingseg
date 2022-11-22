import argparse
import os
from glob import glob
from itertools import chain, compress
from shutil import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.ndimage import median_filter
from scipy.special import kl_div  # pylint: disable=no-name-in-module
from tqdm import tqdm

from dataset import image_label_paths_from_dir
from preprocess import strided_slices
from utils import load_colormap, suppress_tensorflow_warnings


suppress_tensorflow_warnings()


### Generation of synthetic samples from overlapping masks

def cropped_panel_path(cropped_dir: str,
                       panel_name: str):
    im_path = glob(os.path.join(cropped_dir, f"{panel_name}*"))
    assert len(im_path) == 1, f"Not only one panel image found for {im_path}"
    
    return im_path[0]


def load_sorted_patch_masks(patch_masks: list[str],
                            max_num_patches: int):
    # Sort patches by index ("*_<idx>.png") 
    patch_index = lambda x: int(os.path.splitext(x)[0].split("_")[-1])
    patch_masks.sort(key=patch_index)
        
    # Fill missing patches
    if max_num_patches > len(patch_masks):
        missing = sorted(set(range(max_num_patches)) - set(map(patch_index, patch_masks)))
        for miss in missing:
            patch_masks.insert(miss, None)

    # Load patch masks
    patch_masks = [(Image.open(p) if p is not None else None) for p in patch_masks]
    return patch_masks


def merge_patch_masks(patch_masks: list[Image.Image], 
                      patch_positions: np.ndarray,
                      panel_size: tuple[int, int]) -> Image.Image:
    # Combine patches to 4-channel image and reduce with tf.argmax(tf.bincount)
    segmask = np.zeros(panel_size[::-1] + (4,), dtype=np.uint8)

    # Grid indices
    # 0: top left, 1: top right, 2: bottom left, 3: bottom right
    x = np.unique(patch_positions[:,0], return_inverse=True)[1]
    y = np.unique(patch_positions[:,1], return_inverse=True)[1]
    grid_inds = x % 2 + 2 * (y % 2)

    for patch, box, idx in zip(patch_masks, patch_positions, grid_inds):
        if patch is None: continue
        
        patch = np.asarray(patch)
        # (x1, x2, y1, y2)
        segmask[box[1]:box[3], box[0]:box[2], idx] = patch

    # Most frequent value wins but empty stays empty
    empty_pixels = ~np.any(segmask > 0, axis=-1)
    segmask = np.reshape(segmask, (-1, segmask.shape[-1])).astype(int)
    segmask = tf.math.bincount(segmask, axis=-1)[:,1:]
    segmask = tf.math.argmax(segmask, axis=-1) + 1
    segmask = np.reshape(segmask.numpy(), panel_size[::-1]).astype(np.uint8)
    segmask[empty_pixels] = 0
    segmask = median_filter(segmask, 3) # smoothen

    # Array to PIL Image
    palette = next(p for p in patch_masks if p is not None).getpalette()
    segmask = Image.fromarray(segmask, mode="P")
    segmask.putpalette(palette)

    return segmask


def generate_synthetic_samples(data_dir: str):
    annotated_dir = os.path.join(data_dir, "annotated")
    assert os.path.isdir(annotated_dir)
    cropped_dir = os.path.join(data_dir, "cropped")
    assert os.path.isdir(cropped_dir)
    composed_dir = os.path.join(data_dir, "composed")
    os.makedirs(composed_dir, exist_ok=True)

    # Delete previously generated synthetic samples
    for im_path in chain(*image_label_paths_from_dir(annotated_dir)):
        if "_syn_" in im_path:
            os.remove(im_path)

    # Collect paths
    _, label_paths = image_label_paths_from_dir(annotated_dir)
    print(len(label_paths), "annotated images found for sample generation")

    # Filter out oversampled and synthetic images
    mask = ["_over_" not in p and "_syn_" not in p for p in label_paths]
    label_paths = label_paths[mask]

    # Assign panel name path to patches
    panel_names = [os.path.splitext(os.path.basename(p))[0].split("_")[0] for p in label_paths]
    df = pd.DataFrame.from_dict({"patch_mask": label_paths, "panel_name": panel_names})

    # Only images with at least four labeled patches
    df = df[df.groupby("panel_name")["panel_name"].transform('size') >= 4]
    
    # Generate synthetic samples
    for panel_name in tqdm(pd.unique(df["panel_name"])):
        # Load panel image and slice positions
        panel = Image.open(cropped_panel_path(cropped_dir, panel_name))
        slices = strided_slices(*panel.size)

        # Crop thight and shift slices
        # box = tuple(slices[:,:2].min(axis=0)) + tuple(slices[:, 2:].max(axis=0))
        # im = im.crop(box)
        # slices -= 2 * box[:2]

        # Load and merge patch masks
        patch_masks = df[df["panel_name"] == panel_name]["patch_mask"].tolist()
        patch_masks = load_sorted_patch_masks(patch_masks, len(slices))
        segmask = merge_patch_masks(patch_masks, slices, panel.size)

        # Save merged image
        root = os.path.join(composed_dir, panel_name)
        segmask.save(f"{root}.png")
        panel.save(f"{root}.jpg")

        ### Generate new patches
        # Crop image and mask to create centered slices
        # box = (1, 1, panel.width - 1, panel.height - 1)
        # panel = panel.crop(box)
        # segmask = segmask.crop(box)

        root = os.path.join(annotated_dir, panel_name)
        for idx, box in enumerate(slices):
            # Crop to patch
            patch_mask = segmask.crop(box)
            patch_image = panel.crop(box)
            
            # Skip partly empty
            empty_pixels = np.asarray(patch_mask) == 0
            if np.count_nonzero(empty_pixels) / empty_pixels.size > 0.01: 
                continue
            
            # Save generated patch
            name = f"{root}_syn_{idx}"
            patch_image.save(f"{name}.jpg")
            patch_mask.save(f"{name}.png")

### Training validation split

def plot_distributions(train_dist: np.ndarray,
                       val_dist: np.ndarray,
                       full_dist: np.ndarray,
                       classes: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.3
    x = np.arange(len(classes))

    # Distributions
    ax1.bar(x - width, train_dist * 100, width, label="Training")
    ax1.bar(x, val_dist * 100, width, label="Validation")
    ax1.bar(x + width, full_dist * 100, width, label="Full")
    ax1.set_ylabel("Frequency [%]")

    # Errors
    val_error   = (val_dist - full_dist)/full_dist
    train_error = (train_dist - full_dist)/full_dist

    ax2.bar(x - width/2, train_error * 100, width, label="Training")
    ax2.bar(x + width/2, val_error * 100, width, label="Validation")
    ax2.set_ylabel("Relative Error [%]")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation="vertical")
        ax.legend()

    return fig


def plot_ga_history(history):
    fig, ax = plt.subplots()

    ax.plot(history)

    ax.set_yscale("log")
    ax.set_ylim(None, 1e-3)
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Epoch")
    return fig


def calculate_probabilities(lbl_paths, num_classes, drop_zero=False, average=True):
    '''Calculate the pixelwise class weight from a list of indexed images
    '''
    assert len(lbl_paths) > 0
    assert all(os.path.isfile(p) for p in lbl_paths)
    assert isinstance(num_classes, int) and num_classes > 0

    bins = []

    for path in lbl_paths:
        label = np.asarray(Image.open(path)).astype(int).ravel()
        assert label.max() < num_classes
        bins.append(np.bincount(label, minlength=num_classes)[:num_classes])

    bins = np.asarray(bins)
    if average:
        bins = np.sum(bins, axis=0)

    if drop_zero:
        bins = bins[..., 1:]

    return np.nan_to_num(bins / bins.sum(axis=-1, keepdims=True), posinf=0.) # percentage


def dists_from_mask(val_mask: np.ndarray, probs: np.ndarray):
    train_dist = probs[~val_mask].mean(axis=0)
    val_dist   = probs[val_mask].mean(axis=0)

    return train_dist, val_dist


def fitness(val_mask: np.ndarray, probs: np.ndarray, aimed_val_ratio: float):
    train_dist, val_dist = dists_from_mask(val_mask, probs)
    fit = np.sum(kl_div(train_dist, val_dist))

    # Distance from desired validation set size 
    d = np.abs(np.sum(val_mask)/len(val_mask) - aimed_val_ratio)
    if d > 0.01: fit += 1 + d # penality

    return fit


def validation_split(data_dir: str,
                     aimed_val_ratio: float = 0.2,
                     split_epochs: int = 5_000):
    assert 0 < aimed_val_ratio < 1

    # Possible classes to consider for split
    classes = list(load_colormap(data_dir)[0].keys())[1:]

    # Collect paths
    annotated_dir = os.path.join(data_dir, "annotated")
    assert os.path.isdir(annotated_dir), f"{annotated_dir} not found"

    image_paths, label_paths = image_label_paths_from_dir(annotated_dir)
    num_samples = len(image_paths)
    print(num_samples, "annotated images found for splitting")

    # Full data distribution before split
    probs = calculate_probabilities(label_paths, len(classes) + 1, drop_zero=True, average=False)
    full_dist = probs.mean(axis=0)

    # Genetic algorithm for optimal split
    # Initial population
    pop_size = 100
    pop = np.zeros((pop_size, num_samples), dtype=bool)
    pop[:,:int(num_samples * aimed_val_ratio)] = True
    pop = np.apply_along_axis(np.random.permutation, -1, pop)

    history = []
    for _ in tqdm(range(split_epochs), desc="Genetic algorithm"):
        # Fitness evaluation
        pop_fit = np.apply_along_axis(fitness, -1, pop, probs, aimed_val_ratio)

        # Tournament selection
        k = 3
        tournaments = np.random.randint(0, pop_size, (pop_size - 1, k))
        winners = np.argmin(pop_fit[tournaments], axis=-1)
        winners = np.take_along_axis(tournaments, winners[...,None], axis=-1)[:,0]
        
        # Generate offsprings from tournament winners
        offsprings = pop[winners]
        # Mutate with standard bit-flip (p=1/n)
        mutation = np.random.binomial(1, 1/offsprings.shape[1], size=offsprings.shape).astype(bool)
        offsprings[mutation] = ~offsprings[mutation]

        # Copy elitist and replace parents
        elite = pop[np.argmin(pop_fit)]
        pop = np.append(offsprings, elite[None], axis=0)

        history.append(np.min(pop_fit))

    print(f"Best fitness: {history[-1]:.3g}")
    print(f"Validation set size: {np.count_nonzero(elite)/num_samples * 100:.1f}")

    # Plot and save split resuts
    fig = plot_ga_history(history)
    save_path = os.path.join(data_dir, "ga_history.png")
    fig.savefig(save_path, pad_inches=0.05, bbox_inches="tight")
    fig = plot_distributions(*dists_from_mask(elite, probs), full_dist, classes)
    save_path = os.path.join(data_dir, "train_val_distributions.png")
    fig.savefig(save_path, pad_inches=0.05, bbox_inches="tight")

    # Move images
    train_dir = os.path.join(data_dir, "training")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(data_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    for dst_dir, mask in ((val_dir, elite), (train_dir, ~elite)):
        for im in chain(compress(label_paths, mask),compress(image_paths, mask)):
            save_path = os.path.join(dst_dir, os.path.basename(im))
            os.replace(im, save_path)

### Oversampling of training data

def plot_class_distribution(dist: np.ndarray, classes: list):
    fig, ax = plt.subplots()

    x = np.arange(len(dist))
    ax.bar(x, dist * 100)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel("Frequency [%]")
    return fig


def oversample_train_set(data_dir: str):
    train_dir = os.path.join(data_dir, "training")
    assert os.path.isdir(train_dir)

    # Possible classes to consider for oversampling
    classes = list(load_colormap(data_dir)[0].keys())[1:]

    # Revert previous oversampling
    for im_path in chain(*image_label_paths_from_dir(train_dir)):
        if "_over_" in im_path:
            os.remove(im_path)

    # Collect paths
    image_paths, label_paths = image_label_paths_from_dir(train_dir)
    num_samples = len(label_paths)
    print(num_samples, "training images found for oversampling")

    # Data distribution before oversampling
    probs = calculate_probabilities(label_paths, len(classes) + 1, drop_zero=True, average=False)
    dist_before = probs.mean(axis=0)

    print("Before oversampling")
    print(f"min-max ratio: {dist_before.max() / dist_before.min():.2g}")
    print(f"std:           {dist_before.std():.2g}")
    fig = plot_class_distribution(dist_before, classes)
    save_path = os.path.join(data_dir, "before_oversampling.png")
    fig.savefig(save_path, pad_inches=0.05, bbox_inches="tight")

    # Inverse class frequency weights
    weights = dist_before.max() / dist_before

    # Number of additional samples per image
    alpha = 2
    oversampling = probs @ weights
    oversampling = np.maximum(oversampling - oversampling.mean() - alpha, 0)
    oversampling = np.floor(oversampling).astype(int)
    
    print(f"Max oversampling: \t{oversampling.max()}")
    print(f"Mean oversampling:\t{oversampling.mean():.2g}")
    num_add_samples = np.sum(oversampling[oversampling > 0])
    print(f"Additional samples: {num_add_samples} (+{num_add_samples/len(probs) * 100:.2g} %)")

    # Data distribution after oversampling
    dist_after = (oversampling + 1) @ probs / np.sum(oversampling + 1)

    print("After oversampling")
    print(f"min-max ratio: {dist_after.max() / dist_after.min():.2g}")
    print(f"std:           {dist_after.std():.2g}")
    fig = plot_class_distribution(dist_after, classes)
    save_path = os.path.join(data_dir, "after_oversampling.png")
    fig.savefig(save_path, pad_inches=0.05, bbox_inches="tight")

    # Copy images
    for *imgs, num_samples in zip(image_paths, label_paths, oversampling):
        if num_samples == 0: 
            continue
        
        for im_path in imgs:
            root, ext = os.path.splitext(im_path)
            for idx in range(num_samples):
                copy_path = f"{root}_over_{idx}{ext}"
                copy(im_path, copy_path)
    

def main(data_dir: str,
         syn_samples = None,
         train_val_split = None,
         aimed_val_ratio = None,
         split_epochs = None,
         oversample_train = None):

    print("Please ensure that image patches and labels are stored in data_dir/annotated")
    # Generate synthetic samples (needs masks)
    if syn_samples:
        print("Generating synthetic samples...")
        generate_synthetic_samples(data_dir)

    # Validation split
    if train_val_split:
        print("Splitting into training and validation set...")
        validation_split(data_dir, aimed_val_ratio=aimed_val_ratio, split_epochs=split_epochs)

    # Oversampling (needs masks)
    if oversample_train:
        print("Oversampling training set...")
        oversample_train_set(data_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data",
        help="Directory with the 'annotated' patches and 'cropped' panel images")

    parser.add_argument("--syn_samples", action='store_true', default=False,
        help="Generate synthetic samples from overlapping segmentation masks")

    parser.add_argument("--train_val_split", action='store_true', default=False,
        help="Split annotated patches into training and validation set")
    parser.add_argument("--aimed_val_ratio", type=float, default=0.2,
        help="Desired fraction of the validation set")
    parser.add_argument("--split_epochs", type=int, default=5_000,
        help="Evolution epochs of the genetic algorithm for training validation split")

    parser.add_argument("--oversample_train", action='store_true', default=False,
        help="Oversample training data by class frequency")

    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**parse_args())
