import argparse
import os
from glob import glob
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm


# Old border_h=0.2
def image_crop_box(im_width, im_height, border_w=0.02, border_h=0.25) -> Tuple[int, int, int, int]:
    '''Crop image and remove defined borders'''
    assert isinstance(border_w, float) and 0 <= border_w < 1
    assert isinstance(border_h, float) and 0 <= border_h < 1
    x1 = int(border_w * im_width) // 2
    y1 = int(border_h * im_height) // 2
    w = int((1 - border_w) * im_width)
    h = int((1 - border_h) * im_height)
    return (x1, y1, x1 + w, y1 + h)


def enhance_image(im: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
    '''Returns contrast enhanced and sharpened image. Image must be in [0, 255]
    '''
    notPIL = not isinstance(im, Image.Image)
    if notPIL: # ndarray -> PIL
        dtype = im.dtype
        im = Image.fromarray(im.astype(np.uint8))

    im = ImageOps.autocontrast(im, cutoff=0.25)
    im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))

    if notPIL: # PIL -> ndarray
        im = np.asarray(im).astype(dtype)
    return im


def strided_slices(w, h, slice_size=384, slice_overlap=64, center=True):
    '''Returns box coordinates in PIL order (x1, y1, x2, y2)'''
    stride = slice_size - slice_overlap
    slices = [(x, y, x + slice_size, y + slice_size) # Slice with stride (overlap)
        for x in range(0, w - slice_size + 1, stride) for y in range(0, h - slice_size + 1, stride)]
    slices = np.asarray(slices, dtype=int)
    if center: # Spread boxes from image center
        delta = ([w, h] - slices[:, 2:].max(axis=0))//2
        slices += 2 * [*delta]
    return slices


def preprocess_images(
    images_dir: str,
    dst_dir: str,
    im_height: int = 2752,
    slices: bool = False):
    '''Crop, resize, enhance, and slice (optional) images for training or inference
    '''
    assert os.path.isdir(dst_dir)

    # Organize paths
    image_paths = glob(os.path.join(images_dir, "**/*.JPG"), recursive=True)
    assert len(image_paths) > 0, "no images found"

    cropped_dir = os.path.join(dst_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    if slices:
        sliced_dir = os.path.join(dst_dir, "sliced")
        os.makedirs(sliced_dir, exist_ok=True)

    # Calculate median image shape for resizing
    wh = np.median([Image.open(im).size for im in image_paths], axis=0)
    box = image_crop_box(*wh)
    wh = np.asarray((box[2] - box[0], box[3] - box[1]))
    wh = wh * (im_height / wh[1])
    wh = (np.round(wh / 64) * 64).astype(int)
    print("Resulting image size:", wh)

    for im_path in tqdm(image_paths):
        # Preprocessing operations
        im = Image.open(im_path)
        im = im.crop(box=image_crop_box(*im.size))
        im = im.resize(wh, Image.LANCZOS)
        im = enhance_image(im)

        # Save cropped image
        root, _ = os.path.splitext(os.path.basename(im_path))
        im.save(os.path.join(cropped_dir, f"{root}.JPG"))

        # Slice to patches
        if slices:
            root = os.path.join(sliced_dir, root)
            for idx, box in enumerate(strided_slices(*im.size)):
                name = f"{root}_{idx}.JPG"
                im.crop(box).save(name, optimize=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default="data/panels",
        help="Root directory where the panel images are stored (in subfolders)")
    parser.add_argument("--dst_dir", type=str, default="data",
        help="Destination for the preprocessed images and slices")
    parser.add_argument("--slices", action='store_true', default=False,
        help="Slice images after preprocessing to patches for training")

    return vars(parser.parse_args())


if __name__ == "__main__":
    preprocess_images(**parse_args())
