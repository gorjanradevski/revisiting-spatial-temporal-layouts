import argparse


class Parser:
    def __init__(self, parser_description: str):
        self.parser = argparse.ArgumentParser(description=parser_description)
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="The name of the dataset, either something or action_genome",
        )
        self.parser.add_argument(
            "--log_filepath",
            type=str,
            default=None,
            help="Where to log the progress.",
        )
        self.parser.add_argument(
            "--train_dataset_path",
            type=str,
            default=None,
            help="Path to the train dataset.",
        )
        self.parser.add_argument(
            "--val_dataset_path",
            type=str,
            default=None,
            help="Path to the val dataset.",
        )
        self.parser.add_argument(
            "--test_dataset_path",
            type=str,
            default=None,
            help="Path to the test dataset.",
        )
        self.parser.add_argument(
            "--labels_path",
            type=str,
            default=None,
            help="Path to the labels.",
        )
        self.parser.add_argument(
            "--videoid2size_path",
            type=str,
            default="data/videoid2size.json",
            help="Path to the videoid2size json file.",
        )
        self.parser.add_argument(
            "--layout_num_frames",
            type=int,
            default=16,
            help="The number of layout frames to sample per video.",
        )
        self.parser.add_argument(
            "--score_threshold",
            type=float,
            default=0.5,
            help="The score threshold for the categories.",
        )
        self.parser.add_argument(
            "--num_spatial_layers",
            type=int,
            default=4,
            help="The number of spatial transformer layers.",
        )
        self.parser.add_argument(
            "--num_temporal_layers",
            type=int,
            default=8,
            help="The number of temporal transformer layers.",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="The batch size.",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-5,
            help="The learning rate.",
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-3,
            help="The weight decay.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="The number of processor workers.",
        )
        self.parser.add_argument(
            "--clip_val",
            type=float,
            default=5.0,
            help="The gradient clipping value.",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=20,
            help="The number of epochs to train the model.",
        )
        self.parser.add_argument(
            "--warmup_epochs",
            type=int,
            default=2,
            help="The number warmup epochs.",
        )
        self.parser.add_argument(
            "--save_model_path",
            type=str,
            default="models/best.pt",
            help="Where to save the model.",
        )
        self.parser.add_argument(
            "--save_backbone_path",
            type=str,
            default=None,
            help="Where to save the STLT backbone.",
        )
        self.parser.add_argument(
            "--load_backbone_path",
            type=str,
            default=None,
            help="From where to load the STLT backbone.",
        )
        self.parser.add_argument(
            "--freeze_backbone",
            action="store_true",
            help="Whether to freeze the backbone.",
        )
        self.parser.add_argument(
            "--features_path",
            type=str,
            default=None,
            help="Whether to use video features.",
        )
        self.parser.add_argument(
            "--checkpoint_path",
            type=str,
            default="models/best.pt",
            help="Checkpoint to a trained model.",
        )

    def parse_args(self):
        return self.parser.parse_args()
