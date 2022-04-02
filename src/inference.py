import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling.datasets import DataConfig, collaters_factory, datasets_factory
from modelling.configs import model_configs_factory
from modelling.models import models_factory
from utils.evaluation import evaluators_factory
from utils.parser import Parser
from utils.train_inference_utils import get_device, move_batch_to_device


@torch.no_grad()
def inference(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = get_device(logger=logging.getLogger(__name__))
    logging.info("Preparing dataset...")
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.test_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        layout_num_frames=args.layout_num_frames,
        appearance_num_frames=args.appearance_num_frames,
        videos_path=args.videos_path,
        train=False,
    )
    test_dataset = datasets_factory[args.dataset_type](data_config)
    num_samples = len(test_dataset)
    logging.info(f"Inference on {num_samples}")
    collater = collaters_factory[args.dataset_type](data_config)
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
    model_config = model_configs_factory[args.model_name](
        num_classes=num_classes,
        unique_categories=len(data_config.category2id),
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        appearance_num_frames=args.appearance_num_frames,
        resnet_model_path=args.resnet_model_path,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    model = models_factory[args.model_name](model_config).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except RuntimeError as e:
        logging.warning(
            "Default loading failed, loading with strict=False. If it's only "
            "score_embedding modules it's ok. Otherwise see exception below"
        )
        logging.warning(e)
        model.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device), strict=False
        )
    model.train(False)
    logging.info("Starting inference...")
    evaluator = evaluators_factory[args.dataset_name](
        num_samples, num_classes, model.logit_names
    )
    for batch in tqdm(test_loader):
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        evaluator.process(logits, batch["labels"])

    metrics = evaluator.evaluate()
    logging.info("=================================")
    logging.info("The metrics are:")
    for m in metrics.keys():
        logging.info(f"{m}: {round(metrics[m] * 100, 2)}")
    logging.info("=================================")


def main():
    parser = Parser("Inference with a model, currenly STLT, LCF, CAF, and CACNF.")
    inference(parser.parse_args())


if __name__ == "__main__":
    main()
