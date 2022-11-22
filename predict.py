import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tqdm import tqdm

from dataset import load_dataset
from models import make_feature_extractor
from utils import allow_gpu_memory_growth, indexed_from_arr, load_colormap


def load_model(model_path: str) -> Model:
    assert os.path.isfile(model_path)
    return tf.keras.models.load_model(model_path, compile=False)


def save_segmasks(results_dir: str,
                  masks: np.ndarray, 
                  image_paths: list[str], 
                  palette: list[int]):
    for mask, path in zip(masks, image_paths):
        path = f"{os.path.splitext(os.path.basename(path))[0]}_pred.png"
        path = os.path.join(results_dir, path)

        # Save as indexed color image
        mask = indexed_from_arr(mask, palette)
        mask.save(path)


def save_entropy_maps(results_dir: str,
                      entropies: np.ndarray, 
                      image_paths: list[str]):
    # Rescale to grayscale image range
    entropies = (entropies * 255).astype(np.uint8)

    for emap, path in zip(entropies, image_paths):
        path = f"{os.path.splitext(os.path.basename(path))[0]}_entropy.png"
        path = os.path.join(results_dir, path)

        # Save as indexed color image
        emap = Image.fromarray(emap)
        emap.save(path)


def save_overlay(results_dir: str,
                 images: np.ndarray, 
                 segmasks: np.ndarray, 
                 entropies: np.ndarray, 
                 dists: np.ndarray,
                 image_paths: list[str],
                 classes: dict,
                 palette: list[int]):
    for image, mask, h, dist, path in zip(images, segmasks, entropies, dists, image_paths):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 9),
            gridspec_kw={'width_ratios': [6, 1]})

        # Image mask overlay
        ax1.imshow(image.astype(np.uint8))
        ax1.imshow(indexed_from_arr(mask, palette), alpha=0.4)

        ax1.axis("off")
        ax1.set_title(f"$\\barH={h.mean():.2f}$")

        # Class distribution
        X = np.arange(len(classes))
        colors = np.asarray(palette).reshape((-1, 3)) / 255
        ax2.barh(X, dist * 100, color=colors, alpha=0.8)

        ax2.set_title("[log %]")
        ax2.set_xscale("log")
        ax2.set_xlim(None, 100)
        ax2.xaxis.set_visible(False)

        labels = [l.replace(" ", "\n") for l in list(classes.keys())]
        labels = [f"{name} ({p:3.1f} %)" for name, p in zip(labels, dist * 100)]
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_yticks(X)
        ax2.set_yticklabels(labels)
        for t, c in zip(ax2.get_yticklabels(), colors):
            t.set_color(c)

        fig.tight_layout(pad=1.5, w_pad=3)

        # Save image
        path = f"{os.path.splitext(os.path.basename(path))[0]}_zoverlay.png"
        path = os.path.join(results_dir, path)
        fig.savefig(path, dpi=150)
        plt.close()


def save_side_by_side(results_dir: str,
                      images: np.ndarray,
                      segmasks: np.ndarray, 
                      dists: np.ndarray, 
                      image_paths: list[str],
                      classes: dict,
                      palette: list[int]):
    for image, mask, dist, path in zip(images, segmasks, dists, image_paths):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4),
            gridspec_kw={'width_ratios': [4, 4, 4, 1]})

        # Panel image
        ax1.imshow(image.astype(np.uint8))
        ax1.axis("off")
        
        # Predicted mask
        ax2.imshow(indexed_from_arr(mask, palette))
        ax2.axis("off")

        # Overlayed image
        ax3.imshow(image.astype(np.uint8))
        ax3.imshow(indexed_from_arr(mask, palette), alpha=0.4)
        ax3.axis("off")

        # Colors of classes
        colors = np.asarray(palette).reshape((-1, 3)) / 255
        
        # Bar plot
        X = np.arange(len(classes))
        ax4.barh(X, dist * 100, color=colors)
        
        ax4.set_title("Coverage [%]")
        ax4.set_xlim(None, 100)
        ax4.xaxis.set_visible(False)

        labels = list(classes.keys())
        labels = [l.replace(" ", "\n") for l in labels]
        labels = [f"{name} ({p:3.1f} %)" for name, p in zip(labels, dist * 100)]
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.tick_right()
        ax4.set_yticks(X)
        ax4.set_yticklabels(labels)
        for t, c in zip(ax4.get_yticklabels(), colors):
            t.set_color(c)

        fig.tight_layout()

        # Save image
        path = f"{os.path.splitext(os.path.basename(path))[0]}_analysis.png"
        path = os.path.join(results_dir, path)
        fig.savefig(path, dpi=150)
        plt.close()


def save_statistics(results_dir: str,
                    statistics: np.ndarray,
                    classes: dict):
    # Build statistics dataframe
    df = pd.DataFrame(statistics, columns=["Name"] + list(classes.keys()) + ["Entropy"])
    df = df.drop("empty", axis=1)

    save_dir = os.path.join(results_dir, "ai_coverage.xlsx")
    df.to_excel(save_dir, sheet_name="Data AI", index=False)


def build_dataset(image_paths: list[str], 
                  batch_size: int, 
                  downsample: int) -> tf.data.Dataset:
    dataset = load_dataset(image_paths, downsample=downsample, dynamic_shape=(batch_size == 1))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def predict_embeddings(model_path: str,
                       image_paths: list[str],
                       batch_size: int = 128,
                       downsample: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # Load model and dataset
    feature_extractor = make_feature_extractor(load_model(model_path))
    dataset = build_dataset(image_paths, batch_size, downsample)

    # Predict and normalize embeddings
    embeddings = np.vstack([feature_extractor.predict_on_batch(x) for x in dataset])
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    del feature_extractor

    # Reduce dimensionality for t-SNE with PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(embeddings)
    print(f"Reduced variance to {np.sum(pca.explained_variance_ratio_):.2g}")
    
    # Visualize with t-SNE
    tsne = TSNE(n_components=2, init="pca", perplexity=50, learning_rate=200, n_iter=2000)
    X_tsne = tsne.fit_transform(X_pca)
    print(f"Final KL divergence: {tsne.kl_divergence_:.2f}")

    return X_pca, X_tsne


def predict_images(model_path: str,
                   image_paths: list[str],
                   batch_size: int = 128,
                   downsample: int = 2):
    model = load_model(model_path)
    dataset = build_dataset(image_paths, batch_size, downsample)
    
    for images in dataset:
        # Inference and upsample predictions
        logits = model(images)
        if downsample > 1:
            logits = K.resize_images(logits, downsample, downsample, K.image_data_format(),
                interpolation="bilinear")
            images = K.resize_images(images, downsample, downsample, K.image_data_format(),
                interpolation="bilinear")
        
        logits = logits.numpy()
        images = images.numpy()
        
        # Post-processing
        entropies = entropy(logits, base=logits.shape[-1], axis=-1)
        masks = np.argmax(logits, axis=-1) + 1
        dists = np.apply_along_axis(np.bincount, -1, masks.reshape((masks.shape[0], -1)),
            minlength=logits.shape[-1] + 1)
        dists = dists / dists.sum(axis=-1, keepdims=True)

        yield images, logits, entropies, masks, dists


def predict(data_dir: str,
            results_dir: str,
            model_path: str,
            batch_size: int = 16,
            downsample: int = 2,
            save_entropies: bool = False,
            save_overlay_image: bool = False,
            save_overview_image: bool = False):

    os.makedirs(results_dir, exist_ok=True)
    
    # Collect paths
    image_paths = glob(os.path.join(data_dir, "**/*.JPG"), recursive=True)
    assert len(image_paths) > 0, "no images found"

    classes, palette = load_colormap(os.path.join(data_dir, ".."))
    
    # Infere images and batch assigned paths
    image_paths_batched = [
        image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    preds = predict_images(model_path, image_paths, batch_size=batch_size, downsample=downsample)
    
    statistics = []
    for x, paths in tqdm(zip(preds, image_paths_batched), 
        desc="Predicting...", total=len(image_paths_batched)):

        # Inference
        images, _logits, entropies, masks, dists = x

        # Save result images
        save_segmasks(results_dir, masks, paths, palette)
        if save_entropies:
            save_entropy_maps(results_dir, entropies, paths)
        if save_overlay_image:
            save_overlay(results_dir, images, masks, entropies, dists, paths, classes, palette)
        if save_overview_image:
            save_side_by_side(results_dir, images, masks, dists, paths, classes, palette)

        names = np.vstack(list(map(os.path.basename, paths)))
        statistics.append(
            np.concatenate((names, dists, entropies.mean(axis=(1, 2))[:,None]), axis=-1))

    save_statistics(results_dir, np.vstack(statistics), classes)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/validation", # or data/cropped
        help="Directory with images for inference") 
    parser.add_argument("--results_dir", type=str, default="results/prediction",
        help="Destination for segmentation masks, visualizations, and summarized statistics")
    parser.add_argument("--model_path", type=str, 
        default="results/models/4_final_balanced_efficientnet_unet/model.h5")
    
    parser.add_argument("--batch_size", type=int, default=32) # must be smaller for big images
    parser.add_argument("--downsample", type=int, default=2,
        help="Image downsampling factor from model training")

    parser.add_argument("--save_entropies", action='store_true', default=False,
        help="Save uncertainty map calculated by information entropy of logits")
    parser.add_argument("--save_overlay", action='store_true', default=False, dest="save_overlay_image",
        help="Save input overlayed by segmentation mask and visualized statistics")
    parser.add_argument("--save_overview", action='store_true', default=False, dest="save_overview_image",
        help="Save overview image with input, segmentation mask, and visualized statistics")
    
    return vars(parser.parse_args())


if __name__ == "__main__":
    allow_gpu_memory_growth()
    predict(**parse_args())
