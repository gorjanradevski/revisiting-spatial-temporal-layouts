import argparse
import io
import json
import logging
from typing import List

import h5py
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import resize_boxes


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=True, min_size=240, max_size=540
        )
        self.pooler = nn.AdaptiveAvgPool2d(output_size=(3, 3))

    def forward(self, frames: List[torch.Tensor], boxes: List[torch.Tensor]):
        org_sizes = [frame.shape[-2:] for frame in frames]
        transformed_frames, _ = self.model.transform(frames)
        proposals = [
            resize_boxes(boxes[i], org_sizes[i], transformed_frames.image_sizes[i])
            for i in range(len(boxes))
        ]
        backbone_output = self.model.backbone(transformed_frames.tensors)
        selected_rois = self.model.roi_heads.box_roi_pool(
            backbone_output, proposals, transformed_frames.image_sizes
        )
        pooled_rois = self.pooler(selected_rois).flatten(1)

        return pooled_rois


@torch.no_grad()
def dump_video_frames(args):
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    # Prepare model
    device = torch.device(args.device)
    logging.info("Loading model...")
    model = FeatureExtractor().to(device)
    model.eval()
    # Prepare data
    logging.info("Loading dataset...")
    json_file = json.load(open(args.dataset_path))
    # Start dumping
    logging.info("Dumping features...")
    with h5py.File(
        args.videos_path, "r", libver="latest", swmr=True
    ) as videos, h5py.File(args.save_features_path, "a") as video_features:
        cur_video_featues = set(video_features.keys())
        for index, element in enumerate(json_file):
            video_id = element["id"]
            if video_id in cur_video_featues:
                continue
            frame_ids = natsorted([frame_id for frame_id in videos[video_id].keys()])
            num_frames = min(len(frame_ids), len(element["frames"]))
            video_frames = []
            boxes = []
            boxes_per_image = []
            for frame_index in range(num_frames):
                video_frames.append(
                    transforms.ToTensor()(
                        Image.open(
                            io.BytesIO(
                                np.array(videos[video_id][frame_ids[frame_index]])
                            )
                        )
                    ).to(device)
                )
                h, w = video_frames[0].shape[1:3]
                frame_boxes = [[0, 0, w, h]]
                for box in element["frames"][frame_index]:
                    frame_boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                boxes.append(torch.tensor(frame_boxes, device=device))
                boxes_per_image.append(len(frame_boxes))

            # Obtain model output and save features
            features = model(video_frames, boxes).cpu().split(boxes_per_image, 0)
            grp = video_features.create_group(video_id)
            for frame_index in range(len(features)):
                assert (
                    len(element["frames"][frame_index]) + 1  # Because of [0,0,w,h]
                    == features[frame_index].size()[0]
                )
                grp.create_dataset(
                    f"{frame_index}-frame", data=features[frame_index][0].numpy()
                )
                for box_index in range(1, features[frame_index].size()[0]):
                    grp.create_dataset(
                        f"{frame_index}-frame-{box_index-1}-box",
                        data=features[frame_index][box_index].numpy(),
                    )

            if index % args.print_freq == 0:
                logging.info(f"Current index is {index}")


def main():
    parser = argparse.ArgumentParser(
        description="Dumps per-frame and per-bounding box features from Faster R-CNN."
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        default="data/dataset.hdf5",
        help="From where to load the videos.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/val_dataset_174.json",
        help="Path to a smth-smth dataset.",
    )
    parser.add_argument(
        "--save_features_path",
        type=str,
        default="data/per_box_features",
        help="Where to save the per-frame/per-box features.",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to be used",
    )
    args = parser.parse_args()
    dump_video_frames(args)


if __name__ == "__main__":
    main()
