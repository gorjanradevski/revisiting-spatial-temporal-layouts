import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_models.datasets import StltCollater, StltDataConfig, StltDataset
from layout_models.modelling import Stlt, StltModelConfig
from utils.data_utils import get_device
from utils.evaluation import evaluators_factory
from utils.parser import Parser


@torch.no_grad()
def inference(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = get_device(logger=logging.getLogger(__name__))
    logging.info("Preparing dataset...")
    data_config = StltDataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.test_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        num_frames=args.layout_num_frames,
        train=False,
    )
    test_dataset = StltDataset(data_config)
    num_samples = len(test_dataset)
    logging.info(f"Inference on {num_samples}")
    collater = StltCollater(data_config)
    # Prepare loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    logging.info("Preparing model...")
    # Prepare model
    num_classes = len(test_dataset.labels)
    model_config = StltModelConfig(
        num_classes=num_classes,
        unique_categories=len(data_config.category2id),
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    model = Stlt(model_config).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=device), strict=False
    )
    model.train(False)
    logging.info("Starting inference...")
    evaluator = evaluators_factory[args.dataset_name](num_samples, num_classes)
    for batch in tqdm(test_loader):
        # Move tensors to device
        batch = {key: val.to(device) for key, val in batch.items()}
        logits = model(batch)
        evaluator.process(logits, batch["labels"])

    metrics = evaluator.evaluate()
    logging.info("=================================")
    logging.info("The metrics are:")
    for m in metrics.keys():
        logging.info(f"{m}: {round(metrics[m] * 100, 2)}")
    logging.info("=================================")


def main():
    parser = Parser("Inference with STLT model.")
    inference(parser.parse_args())


if __name__ == "__main__":
    main()
