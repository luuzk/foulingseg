import argparse
import os

import tensorflow as tf
from tensorflow import keras

import callbacks
import losses
import metrics
import models
from dataset import (albumentations_aug, build_datasets, preprocess_not,
                     rand_augment)
from utils import (allow_gpu_memory_growth, debug_output, load_dict_if_path,
                   make_submission_path, save_confusion_matrix_plot,
                   save_model_summary, save_params, save_training_history_plot,
                   set_seeds, suppress_tensorflow_warnings,
                   use_mixed_precision)


suppress_tensorflow_warnings()
set_seeds(42) # For reproducability


def train(
    save_dir: str,
    dataset_kwargs: dict,
    model_kwargs: dict,
    optimizer_kwargs: dict,
    max_epochs: int,
    fine_tuning_layer: str = None,
    fine_tuning: bool = False):

    assert os.path.isdir(save_dir)
    assert isinstance(max_epochs, int) and max_epochs > 0
    assert fine_tuning != "only" or "load_weights" in model_kwargs

    # Central evaluation metric (mIoU)
    miou = metrics.MeanIoU(model_kwargs["num_classes"], average="macro")
    loss = losses.SparseCategoricalCrossentropyDiceLoss()

    (train_dataset, val_dataset, test_dataset, train_steps, val_steps,
        test_steps) = build_datasets(**dataset_kwargs)

    # Keras reset and debug output
    keras.backend.clear_session()
    debug_output((train_dataset, val_dataset, test_dataset), ("train", "val", "test"),
        save_dir=save_dir)

    # Create model
    model = models.model_from_args(**model_kwargs)

    # Save model visualization
    save_model_summary(model, os.path.join(save_dir, "model.txt"))
    keras.utils.plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)

    for phase in range(2 if fine_tuning else 1):
        # Fine-tuning preparation
        if fine_tuning and not phase:
            fine_tune_after = models.layer_index_from_encoder(model, fine_tuning_layer)

        # Fine-tuning phase
        if phase:
            models.unfreeze_encoder(model, fine_tune_after)

        # Build model
        model.compile(optimizer=optimizer_from_args(
                        model=model,
                        fine_tuning=bool(phase),
                        **optimizer_kwargs),
                        loss=loss,
                        metrics=[miou])

        # Load history from pretrained model and show performance
        if "load_weights" in model_kwargs and not phase:
            history_path = os.path.dirname(model_kwargs["load_weights"])
            history_path = os.path.join(history_path, "history.json")
            if os.path.isfile(history_path):
                history, _ = load_dict_if_path(history_path, None)
                history_base = history

            print("Performance of pretrained model:")
            model.evaluate(x=test_dataset, steps=test_steps)
            
            if fine_tuning == "only":
                continue

        # Train model
        initial_epoch = len(history["loss"]) if phase else 0
        history = model.fit(
            x=train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[callbacks.EarlyStopping(
                            patience=30,
                            restore_best_weights=True,
                            baseline=min(history["val_loss"]) if phase else None),
                        keras.callbacks.ReduceLROnPlateau(
                            verbose=1,
                            patience=30,
                            min_delta=1e-4,
                            factor=0.2,
                            min_lr=1e-6),
                        callbacks.SampleVisualization(
                            os.path.join(save_dir, "vis_images"),
                            dataset_kwargs["vis_images"],
                            make_gif=True,
                            downsample=dataset_kwargs.get("downsample"))],
            initial_epoch=initial_epoch,
            epochs=max_epochs + initial_epoch,
            verbose=2).history

        model.save(os.path.join(save_dir, "model.h5"))

        # Concatenate history dicts
        if phase:
            history = {k:history_base[k] + history[k] for k in history.keys()}
        else:
            history_base = history # save for later

    save_params(save_dir, history, name="history.json")

    # Evaluation
    result = model.evaluate(
        x=test_dataset,
        steps=test_steps,
        return_dict=True)
    result["cm"] = miou.total_cm.numpy().astype(int).tolist()

    save_params(save_dir, result, name="metrics.json")

    save_training_history_plot(history, save_dir,
        fine_tuning_epoch=len(history_base["loss"]) if fine_tuning else None)
    save_confusion_matrix_plot(result, save_dir)


def optimizer_from_args(
    optimizer,
    learning_rate = None,
    weight_decay = None,
    model = None,
    fine_tuning = False,
    fine_tuning_lr = None,
    fine_tuning_wd = None,
    **kwargs) -> keras.optimizers.Optimizer:

    # Correct weight decay schedule
    if fine_tuning:
        learning_rate = fine_tuning_lr
        weight_decay = fine_tuning_wd

    if weight_decay is not None:
        assert model is not None, "model needed for correct weight decay schedule"
        def wd():
            lr = model.history.history.get("lr", [learning_rate])[-1]
            return lr/learning_rate * weight_decay
        kwargs["weight_decay"] = wd

    return optimizer(learning_rate=learning_rate, **kwargs)


def main(
    config: str,
    data_dir: str,
    results_dir: str,
    batch_size: int = 16,
    max_epochs: int = 300):

    print("TensorFlow version:", tf.__version__)
    allow_gpu_memory_growth()
    use_mixed_precision()

    num_classes = 10
    transforms = rand_augment(1, 2)

    dataset = {
        "aug_func": [albumentations_aug(transforms)],
        "transforms": transforms,
        "preproc_func": [preprocess_not],
        "training_dir": os.path.join(data_dir, "training"),
        "validation_dir": os.path.join(data_dir, "validation"),
        "downsample": 2,
        "batch_size": batch_size,
        "vis_images": os.path.join(data_dir, "vis_images")
    }

    training = {
        "max_epochs": max_epochs,
    }

    optimizer = {
        "optimizer": keras.optimizers.Adam,
        "learning_rate": 1e-3,
        "fine_tuning_lr": 1e-4,
    }

    if config == "baseline":
        model = {
            "func": models.baseline_unet,
            "encoder_filters": 16,
            "encoder_freeze": False,
            "stages": 6,

            "decoder_filters": 16 * 2 ** 4,
            "upsampling": "bilinear",
            "num_classes": num_classes,
        }
    elif config in {"pretrained", "decoder_links", "final"}:
        model = {
            "func": models.efficientnet_unet,
            "b": 2,

            "num_classes": num_classes,
            "decoder_filters": 256,
        }
        training["fine_tuning"] = True
        training["fine_tuning_layer"] = "block2c_add"

        if config in {"decoder_links", "final"}:
            model["decoder_skips"] = True

        if config == "final":
            model["decoder_filters"] = 192
            model["decoder_filter_decay"] = 1.5
            model["squeeze_excitate"] = True
    else:
        raise ValueError("Parameter configuration defined in 'config' not found")

    # Make save directory and save submission parameters
    save_dir = make_submission_path(os.path.join(results_dir, "models"), model["func"], config)
    save_params(save_dir, training, dataset, model, optimizer, name="submission.txt")

    train(save_dir, dataset, model, optimizer, **training)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, 
        choices=["baseline", "pretrained", "decoder_links", "final"], default="final",
        help="Configuration for training (cf., paper, Table 1)")
    parser.add_argument("--data_dir", type=str, default="data",
        help="Directory with training, validation, and visualization image patches")
    parser.add_argument("--results_dir", type=str, default="results",
        help="Results directory for trained models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=300)

    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**parse_args())
