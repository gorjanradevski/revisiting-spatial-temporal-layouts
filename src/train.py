import logging
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling.datasets import DataConfig, datasets_factory, collaters_factory
from modelling.models import models_factory
from modelling.model_configs import model_configs_factory
from utils.train_inference_utils import get_device
from utils.evaluation import evaluators_factory
from utils.parser import Parser
from utils.train_inference_utils import (
    add_weight_decay,
    get_linear_schedule_with_warmup,
)


def train(args):
    if args.log_filepath:
        # Set up logging
        if os.path.exists(args.log_filepath):
            raise ValueError(f"There is a log at {args.log_filepath}!")
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = get_device(logger=logging.getLogger(__name__))
    # Prepare datasets
    logging.info("Preparing datasets...")
    # Prepare train dataset
    train_data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.train_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        layout_num_frames=args.layout_num_frames,
        appearance_num_frames=args.appearance_num_frames,
        videos_path=args.videos_path,
        train=True,
    )
    train_dataset = datasets_factory[args.dataset_type](train_data_config)
    num_training_samples = len(train_dataset)
    # Prepare validation dataset
    val_data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.val_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        layout_num_frames=args.layout_num_frames,
        appearance_num_frames=args.appearance_num_frames,
        videos_path=args.videos_path,
        train=False,
    )
    val_dataset = datasets_factory[args.dataset_type](val_data_config)
    num_validation_samples = len(val_dataset)
    num_classes = len(val_dataset.labels)
    logging.info(f"Training on {num_training_samples}")
    logging.info(f"Validating on {num_validation_samples}")
    # Prepare collaters
    train_collater = collaters_factory[args.dataset_type](train_data_config)
    val_collater = collaters_factory[args.dataset_type](val_data_config)
    # Prepare loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=val_collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    logging.info("Preparing model...")
    # Prepare model
    model_config = model_configs_factory[args.model_name](
        num_classes=num_classes,
        appearance_num_frames=args.appearance_num_frames,
        unique_categories=len(val_data_config.category2id),
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        load_backbone_path=args.load_backbone_path,
        freeze_backbone=args.freeze_backbone,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    model = models_factory[args.model_name](model_config).to(device)
    # Prepare loss and optimize. Hack for the loss but easy :(
    criterion = (
        nn.CrossEntropy()
        if args.dataset_name == "something"
        else nn.BCEWithLogitsLoss()
    )
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(parameters, lr=args.learning_rate)
    num_batches = len(train_dataset) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_epochs * num_batches,
        num_training_steps=args.epochs * num_batches,
    )
    evaluator = evaluators_factory[args.dataset_name](
        num_validation_samples, num_classes
    )
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        # Training loop
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                # Remove past gradients
                optimizer.zero_grad()
                # Move tensors to device
                batch = {
                    key: val.to(device) if isinstance(val, torch.Tensor) else val
                    for key, val in batch.items()
                }
                # Obtain outputs
                logits = model(batch)
                # Measure loss and update weights
                loss = criterion(logits, batch["labels"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                # Update the scheduler
                scheduler.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
        # Validation loop
        model.train(False)
        evaluator.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = {
                    key: val.to(device) if isinstance(val, torch.Tensor) else val
                    for key, val in batch.items()
                }
                logits = model(batch)
                evaluator.process(logits, batch["labels"])
        # Saving logic
        metrics = evaluator.evaluate()
        if evaluator.is_best():
            logging.info("=================================")
            logging.info(f"Found new best on epoch {epoch+1}!")
            logging.info("=================================")
            torch.save(model.state_dict(), args.save_model_path)
            if args.save_backbone_path:
                torch.save(model.backbone.state_dict(), args.save_backbone_path)
        for m in metrics.keys():
            logging.info(f"{m}: {round(metrics[m] * 100, 2)}")


def main():
    parser = Parser("Trains a model.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
