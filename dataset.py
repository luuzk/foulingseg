import os
import random
from glob import glob
from typing import Tuple, Union

import albumentations as A
import numpy as np
import tensorflow as tf
from PIL import Image


### Path operations

def image_label_paths_from_dir(data_dir: Union[str, list[str]], 
                               filter_unlabeled: bool = True):
    '''Returns two lists containing paths of existing .jpg images and .png labels in data_dir.
    data_dir may be a list of directories
    '''
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    assert all(os.path.isdir(d) for d in data_dir), f"{data_dir} must be existing directory"
    
    image_paths = np.asarray([j for d in data_dir for j in glob(os.path.join(d, "*.JPG"))])
    label_paths = np.asarray([os.path.splitext(im)[0] + ".png" for im in image_paths])
    
    # Filter
    label_exists = np.asarray([os.path.isfile(lbl) for lbl in label_paths])
    if filter_unlabeled:
        assert np.count_nonzero(label_exists), "no matching .jpg/.png pairs found"
        return image_paths[label_exists], label_paths[label_exists]
    else:
        return image_paths, label_paths, label_exists

### Dataset loading operations

def load_dataset(image_paths, downsample=None, dynamic_shape=False) -> tf.data.Dataset:
    '''Loads images as tf.data.Dataset. 
    Assumes images of constant shape and same file type.
    '''
    assert len(image_paths) > 0
    assert all(os.path.isfile(path) for path in image_paths)
    assert len(set(os.path.splitext(p)[1].lower() for p in image_paths)) == 1, \
        "image_paths contains more than one file type"
    assert downsample is None or int(downsample) >= 1
    downsample = int(downsample)

    # Workaroud: Use first image as a reference
    ref_im = Image.open(image_paths[0])
    mode   = ref_im.mode
    shape  = np.asarray(ref_im).shape
    if len(shape) == 2: shape += (1,) # (w, h, 1)
    if dynamic_shape: shape = (None, None) + shape[-1:]

    def _parse(path):
        if mode == "P": # fix tensorflow issue 28256 FIXME: TF > 2.3
            load_image_np = lambda p: np.asarray(Image.open(p))[...,None]
            image = tf.numpy_function(load_image_np, [path], tf.uint8)
            image = tf.cast(image, tf.int32)
        else:
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image)
            image = tf.cast(image, tf.float32)
            
        # image = tf.ensure_shape(image, shape) # fix tensorflow issue 24520
        image.set_shape(shape) # fix tensorflow issue 24520

        if downsample > 1:
            method = "nearest" if mode == "P" else "lanczos3"
            new_shape = tf.shape(image)[:2] // downsample
            if method == "nearest" and tf.math.reduce_min(image) == 0: # sparse candidate
                if tf.math.count_nonzero(image, dtype=tf.int32) / tf.size(image) < 5e-4:
                    # Pad points in sparse mask before nearest-neighbour downsampling
                    image = tf.stack([image])
                    image = tf.nn.max_pool2d(image, downsample, 1, "SAME") # 4D tensor only
                    image = tf.squeeze(image, axis=0)
            image = tf.image.resize(image, new_shape, method=method)
            image = tf.clip_by_value(image, 0, 255)

        return image

    dataset = tf.data.Dataset.from_tensor_slices(list(image_paths)) # fix TF issue #20481
    dataset = dataset.map(_parse, tf.data.experimental.AUTOTUNE)
    return dataset


def load_dataset_np(image_paths, downsample=None, return_gen=False) -> np.ndarray:
    '''Loads images as ndarray or generator yielding ndarrays
    '''
    ds_iter = load_dataset(image_paths, downsample=downsample).as_numpy_iterator()
    
    if return_gen:
        return ds_iter
    else:
        return np.asarray(list(ds_iter))


def load_labeled_dataset(img_paths, label_paths, downsample=None) -> tf.data.Dataset:
    '''Load images and labels as a zipped tf.data.Dataset
    '''
    assert len(img_paths) == len(label_paths)

    images = load_dataset(img_paths, downsample)
    labels = load_dataset(label_paths, downsample)
    return tf.data.Dataset.zip((images, labels))

### Combined dataset loading for training

def concat_funcs(funcs):
    '''Concatenates (image, label) transformation functions. First function is applied first to data
    '''
    assert isinstance(funcs, (tuple, list))
    assert all(callable(f) for f in funcs)

    # Concat only if > 1 function
    if len(funcs) == 1:
        return funcs[0]

    def _concatenated(image: tf.Tensor, label: tf.Tensor):
        data = (image, label)
        for func in funcs:
            data = func(*data)
        return data
    return _concatenated


# pylint: disable=unused-argument
def build_datasets(
    training_dir,
    preproc_func,
    aug_func,
    validation_dir=None,
    testing_dir=None,
    batch_size=4,
    max_shuffle_buffer=512,
    downsample=None,
    **kwargs):

    assert isinstance(preproc_func, (list, tuple))
    assert isinstance(aug_func, (list, tuple))
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(max_shuffle_buffer, int) and max_shuffle_buffer > 0

    # Load image paths
    train_paths = list(zip(*image_label_paths_from_dir(training_dir))) # [(image_path, label_path)]
    random.Random(42).shuffle(train_paths)
    if validation_dir:
        train_paths = list(zip(*train_paths))
        val_paths = image_label_paths_from_dir(validation_dir)
    else:
        # Generate holdout validation set (20% of data)
        val_len = len(train_paths) // 5
        assert val_len > 0
        val_paths = list(zip(*train_paths[:val_len]))
        train_paths = list(zip(*train_paths[val_len:]))
    
    if testing_dir:
        test_paths = image_label_paths_from_dir(testing_dir)

    # Build datasets
    # Load and preprocess data
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    aug_func        = concat_funcs(aug_func)
    preproc_func    = concat_funcs(preproc_func)

    # Training dataset
    train_samples = len(train_paths[0])
    buffer_size   = min(max_shuffle_buffer, train_samples)
    train_dataset = load_labeled_dataset(*train_paths, downsample)
    train_dataset = train_dataset.shuffle(buffer_size, seed=42).map(aug_func, AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).map(preproc_func, AUTOTUNE)
    train_dataset = train_dataset.repeat().prefetch(AUTOTUNE)
    
    # Validation dataset
    val_samples = len(val_paths[0])
    val_dataset = load_labeled_dataset(*val_paths, downsample)
    val_dataset = val_dataset.batch(batch_size).map(preproc_func, AUTOTUNE)
    val_dataset = val_dataset.repeat().prefetch(AUTOTUNE)

    # Testing dataset
    if testing_dir: # Load test set only once if given
        test_samples = len(test_paths[0])
        test_dataset = load_labeled_dataset(*test_paths, downsample)
        test_dataset = test_dataset.batch(batch_size).map(preproc_func, AUTOTUNE)
        test_dataset = test_dataset.repeat().prefetch(AUTOTUNE)
    else: # Use validation set otherwise
        test_samples = val_samples
        test_dataset = val_dataset

    # Steps for one epoch
    train_steps = train_samples//batch_size
    val_steps   = val_samples//batch_size
    test_steps  = test_samples//batch_size

    return train_dataset, val_dataset, test_dataset, train_steps, val_steps, test_steps

### Preprocessing methods

def preprocess_asymmetric(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return image/255., label


def preprocess_symmetric(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return (image / 127.5) - 1., label


def preprocess_not(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return image, label

### Augmentations

def rand_augment(n: int, m: int, m_max=10):
    assert n >= 0 and m >= 0

    transforms = A.Compose([
        A.SomeOf([
            # Pixel-level
            A.ToGray(),
            A.HueSaturationValue(60 * m/m_max, 0, 0),
            A.RandomBrightnessContrast(0, 0.6 * m/m_max),
            # Spatial-level
            A.ElasticTransform(192 * m/m_max, 12, 0),
            ClassDropout(int(round(3 * m/m_max))),
            A.CoarseDropout(int(round(24 * m/m_max)), 
                            int(round(24 * m/m_max)), 
                            int(round(24 * m/m_max))), 
                            # mask_fill_value=0),
            A.Compose([
                A.FromFloat(dtype="uint8", max_value=255),
                A.CLAHE(clip_limit=6 * m/m_max, p=1),
                A.ToFloat(max_value=255),
            ])
        ], n, replace=True),
        A.HorizontalFlip(),
        A.Rotate(limit=180, p=1)])

    return transforms


def albumentations_aug(transforms, new_shape=None):
    def _albumentations_numpy(_image: np.ndarray, _mask: np.ndarray):
        augmented = transforms(image=_image, mask=_mask)
        aug_image = np.asarray(augmented["image"]).astype(np.float32)
        aug_label = np.asarray(augmented["mask"]).astype(np.int32)
        return aug_image, aug_label

    def _albumentations_tf(image: tf.Tensor, label: tf.Tensor):
        # Save shapes for restore later
        image_shape = image.get_shape()
        label_shape = label.get_shape()

        if new_shape is not None:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            image_shape = tf.TensorShape(new_shape).concatenate(image_shape[-1:])
            label_shape = tf.TensorShape(new_shape).concatenate(label_shape[-1:])
        
        # Albumentations assumes 2D-labels and images as float in [0, 1]
        image = image / 255.
        label = tf.squeeze(label, axis=-1)

        image, label = tf.numpy_function(_albumentations_numpy, [image, label], (tf.float32, tf.int32))
        image = image * 255.
        image = tf.clip_by_value(image, 0., 255.)
        label = tf.expand_dims(label, axis=-1)

        # Restore shapes
        image = tf.ensure_shape(image, image_shape)
        label = tf.ensure_shape(label, label_shape)
        return image, label
    
    return _albumentations_tf


# pylint: disable=abstract-method
class ClassDropout(A.DualTransform):
    """
    Image and mask augmentation that zeros out mask and image regions corresponding
    to randomly chosen class instances from mask.
    Image can be any number of channels. Mask zero values treated as background. 
    Never drops all foreground classes.
    Inspired by albumentations.MaskDropout
    """

    def __init__(
        self,
        max_objects=1,
        image_fill_value=0,
        mask_fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        """
        Args:
            max_objects: Maximum number of labels that can be zeroed out. Can be tuple, 
            in this case it's [min, max]
            image_fill_value: Fill value to use when filling image.
            mask_fill_value: Fill value to use when filling mask.
        Targets:
            image, mask
        Image types:
            uint8, float32
        """
        super(ClassDropout, self).__init__(always_apply, p)
        self.max_objects = A.to_tuple(max_objects, 1)
        self.image_fill_value = image_fill_value
        self.mask_fill_value = mask_fill_value

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]

        # Find foreground classes
        fg_classes = set(np.unique(mask))
        if 0 in fg_classes:
            fg_classes.remove(0)

        if len(fg_classes) <= 1:
            dropout_mask = None
        else:
            objects_to_drop = random.randint(self.max_objects[0], self.max_objects[1])
            objects_to_drop = min(len(fg_classes) - 1, objects_to_drop) # never drop all fg classes

            class_indices = random.sample(fg_classes, objects_to_drop)
            dropout_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.bool)
            for class_index in class_indices:
                dropout_mask |= mask == class_index

        params.update({"dropout_mask": dropout_mask})
        return params

    def apply(self, img, **params):
        dropout_mask = params["dropout_mask"]

        if dropout_mask is None:
            return img

        img = img.copy()
        img[dropout_mask] = self.image_fill_value

        return img

    def apply_to_mask(self, img, **params):
        dropout_mask = params["dropout_mask"]

        if dropout_mask is None:
            return img

        img = img.copy()
        img[dropout_mask] = self.mask_fill_value
        return img

    def get_transform_init_args_names(self):
        return ("max_objects", "image_fill_value", "mask_fill_value")
