import argparse
import io
import json
import logging

import h5py
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch import nn
from torchvision import resnet152, transforms


class FrameEncoder(nn.Module):
    def __init__(self):
        super(FrameEncoder, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )

    def forward(self, images: torch.Tensor):
        embedded_images = torch.flatten(self.resnet(images), start_dim=1)

        return embedded_images


class FrameTransformer:
    def __init__(self):
        super(FrameTransformer, self).__init__()
        self.transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.CenterCrop((224, 224)),
            ]
        )

    def __call__(self, image: Image):
        return self.transformer(image)


@torch.no_grad()
def dump_video_frames(args):
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    # Prepare model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Loading model...")
    model = FrameEncoder().to(device)
    model.eval()
    # Prepare transforms
    video_transforms = FrameTransformer()
    # Prepare data
    logging.info("Loading dataset...")
    video_ids = json.load(open(args.videoid2size_path))
    with h5py.File(
        args.videos_path, "r", libver="latest", swmr=True
    ) as videos, h5py.File(args.save_features_path, "a") as video_features:
        cur_video_featues = set(video_features.keys())
        # Start dumping
        logging.info("Dumping features...")
        for index, video_id in enumerate(video_ids.keys()):
            if video_id in cur_video_featues:
                continue
            frame_ids = natsorted([frame_id for frame_id in videos[video_id].keys()])
            video_frames = torch.stack(
                [
                    video_transforms(
                        Image.open(io.BytesIO(np.array(videos[video_id][frame_id])))
                    )
                    for frame_id in frame_ids
                ],
                dim=0,
            ).to(device)
            # Obtain model output and save features
            features = model(video_frames).cpu().numpy()
            video_features.create_dataset(video_id, data=features)
            if index % args.print_freq == 0:
                logging.info(f"Current index is {index}")


def main():
    parser = argparse.ArgumentParser(
        description="Dumps perframe features from ResNet152."
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        default="data/dataset.hdf5",
        help="From where to load the videos.",
    )
    parser.add_argument(
        "--videoid2size_path",
        type=str,
        default="data/videoid2size.json",
        help="Path to the videoid2size file.",
    )
    parser.add_argument(
        "--save_features_path",
        type=str,
        default="data/per_frame_features.hdf5",
        help="Where to save the per-frame features.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1000,
        help="How often to print.",
    )
    parser.add_argument(
        "--log_filepath",
        type=str,
        default=None,
        help="The logging destination.",
    )
    args = parser.parse_args()
    dump_video_frames(args)


if __name__ == "__main__":
    main()
