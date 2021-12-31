import json
import re
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from utils.constants import MAX_NUM_OBJECTS, category2id, frame2type
from utils.data_utils import (
    fix_box,
    get_test_layout_indices,
    pad_sequence,
    sample_train_layout_indices,
)


class StltDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        labels: Dict[str, str],
        videoid2size: Dict[str, List[float]],
        num_frames: int = 16,
        train: bool = False,
    ):
        # Load dataset JSON file
        self.json_file = json.load(open(dataset_path))
        self.labels = labels
        self.videoid2size = videoid2size
        self.num_frames = num_frames
        self.train = train

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx: int):
        video_id = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_id]).repeat(2)
        # Start loading
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.num_frames, num_frames)
            if self.train
            else get_test_layout_indices(self.num_frames, num_frames)
        )
        boxes = []
        categories = []
        frame_types = []
        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            frame_types.append(
                frame2type["empty"] if len(frame) == 0 else frame2type["regular"]
            )
            frame_boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0])]
            frame_categories = [category2id["cls"]]
            for element in frame:
                # Prepare box
                box = [element["x1"], element["y1"], element["x2"], element["y2"]]
                # Fix box
                box = fix_box(
                    box, (video_size[1].item(), video_size[0].item())
                )  # Height, Width
                # Transform box
                box = torch.tensor(box) / video_size
                frame_boxes.append(box)
                # Prepare category
                category_name = (
                    "object hand" if "hand" in element["category"] else "object"
                )
                frame_categories.append(category2id[category_name])
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != MAX_NUM_OBJECTS + 1:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(0)
            categories.append(torch.tensor(frame_categories))
            boxes.append(torch.stack(frame_boxes, dim=0))
        # Appending extract element
        # Boxes
        extract_box = torch.full((MAX_NUM_OBJECTS + 1, 4), 0.0)
        extract_box[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes.append(extract_box)
        boxes = torch.stack(boxes, dim=0)
        # Categories
        extract_category = torch.full((MAX_NUM_OBJECTS + 1,), 0)
        extract_category[0] = category2id["cls"]
        categories.append(extract_category)
        categories = torch.stack(categories, dim=0)
        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(frame2type["extract"])
        frame_types = torch.tensor(frame_types)
        # Obtain video label
        video_label = torch.tensor(
            int(self.labels[re.sub("[\[\]]", "", self.json_file[idx]["template"])])
        )

        return (
            video_id,
            categories,
            boxes,
            frame_types,
            length,
            video_label,
        )


class StltCollater:
    def __call__(self, batch):
        (
            _,
            categories,
            boxes,
            frame_types,
            lengths,
            labels,
        ) = zip(*batch)

        # https://github.com/pytorch/pytorch/issues/24816
        # Pad categories
        pad_categories_tensor = torch.full((MAX_NUM_OBJECTS + 1,), 0)
        pad_categories_tensor[0] = category2id["cls"]
        categories = pad_sequence(categories, pad_tensor=pad_categories_tensor)
        # Pad boxes
        pad_boxes_tensor = torch.full((MAX_NUM_OBJECTS + 1, 4), 0.0)
        pad_boxes_tensor[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes = pad_sequence(boxes, pad_tensor=pad_boxes_tensor)
        # Pad frame types
        frame_types = pad_sequence(
            frame_types, pad_tensor=torch.tensor([frame2type["pad"]])
        )
        # Prepare lengths and labels
        lengths = torch.stack([*lengths], dim=0)
        labels = torch.stack([*labels], dim=0)
        # Generate mask for the padding of the boxes
        src_key_padding_mask_boxes = torch.zeros_like(categories, dtype=torch.bool)
        src_key_padding_mask_boxes[torch.where(categories == 0)] = True
        # Generate mask for padding of the frames
        src_key_padding_mask_frames = torch.zeros(frame_types.size(), dtype=torch.bool)
        src_key_padding_mask_frames[
            torch.where(frame_types == frame2type["pad"])
        ] = True

        return {
            "categories": categories,
            "boxes": boxes,
            "frame_types": frame_types,
            "lengths": lengths,
            "src_key_padding_mask_boxes": src_key_padding_mask_boxes,
            "src_key_padding_mask_frames": src_key_padding_mask_frames,
        }, labels
