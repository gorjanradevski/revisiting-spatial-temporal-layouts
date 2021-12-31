import logging
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_models.datasets import (
    StltCollater,
    StltDataConfig,
    StltDataset,
    category2id,
)
from layout_models.modelling import Stlt, StltModelConfig
from utils.data_utils import get_device
from utils.evaluation import Evaluator
from utils.parser import Parser
from utils.train_utils import add_weight_decay, get_linear_schedule_with_warmup


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
    train_data_config = StltDataConfig(
        dataset_path=args.train_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        num_frames=args.layout_num_frames,
        train=True,
    )
    train_dataset = StltDataset(train_data_config)
    # Prepare validation dataset
    val_data_config = StltDataConfig(
        dataset_path=args.val_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        num_frames=args.layout_num_frames,
        train=False,
    )
    val_dataset = StltDataset(val_data_config)
    logging.info(f"Training on {len(train_dataset)}")
    logging.info(f"Validating on {len(val_dataset)}")
    # Identical collating for training and evaluation
    collater = StltCollater(val_data_config)
    # Prepare loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    logging.info("Preparing model...")
    # Prepare model
    model_config = StltModelConfig(
        num_classes=len(train_dataset.labels),
        unique_categories=len(category2id),
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        load_backbone_path=args.load_backbone_path,
        freeze_backbone=args.freeze_backbone,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    model = Stlt(model_config).to(device)
    # Prepare loss and optimizer
    criterion = nn.CrossEntropyLoss()
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(parameters, lr=args.learning_rate)
    num_batches = len(train_dataset) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_epochs * num_batches,
        num_training_steps=args.epochs * num_batches,
    )
    evaluator = Evaluator(len(val_dataset))
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        # Training loop
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for inputs, labels in train_loader:
                # Remove past gradients
                optimizer.zero_grad()
                # Move tensors to device
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)
                # Obtain outputs
                logits = model(inputs)
                # Measure loss and update weights
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                # Update the scheduler
                scheduler.step()
                # Update progress bar
                acc = (logits.argmax(-1) == labels).sum() / labels.size()[0]
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item(), "Acc": acc.item()})

        # Validation loop
        model.train(False)
        evaluator.reset()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                # Move tensors to device
                inputs = {key: val.to(device) for key, val in inputs.items()}
                logits = model(inputs)
                evaluator.process(logits, labels)

        top1_accuracy, top5_accuracy = evaluator.evaluate()
        if evaluator.is_best():
            logging.info("=================================")
            logging.info(
                f"Found new best on epoch {epoch+1} with accuracy "
                f"{round(top1_accuracy * 100, 2)}% (TOP 1), "
                f"{round(top5_accuracy * 100, 2)}% (TOP 5). Saving model!"
            )
            logging.info("=================================")
            torch.save(model.state_dict(), args.save_model_path)
            if args.save_backbone_path:
                torch.save(model.stlt_backbone.state_dict(), args.save_backbone_path)
        else:
            logging.info(
                f"Found accuracy {round(top1_accuracy * 100, 2)}% (TOP 1), "
                f"{round(top5_accuracy * 100, 2)}% (TOP 5) on epoch {epoch+1}"
            )


def main():
    parser = Parser("Trains an STLT model.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
