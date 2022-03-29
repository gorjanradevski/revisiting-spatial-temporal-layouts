import json
import re

import torch
from torch.utils.data import Dataset
from utils.data_utils import (
    fix_box,
    get_test_layout_indices,
    pad_sequence,
    sample_train_layout_indices,
)


class StltDataConfig:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        labels_path: str,
        videoid2size_path: str,
        train: bool,
        **kwargs,
    ):
        assert (
            dataset_name == "something" or dataset_name == "action_genome"
        ), f"{dataset_name} does not exist!"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.videoid2size_path = videoid2size_path
        self.train = train
        self.num_frames = kwargs.pop("num_frames", 16)
        self.max_num_objects = kwargs.pop("max_num_objects", 7)
        self.score_threshold = kwargs.pop("score_threshold", 0.5)
        # Hacking :(
        self.category2id = (
            {
                "pad": 0,
                "hand": 1,
                "object": 2,
                "cls": 3,
            }
            if self.dataset_name == "something"
            else {
                "pad": 0,
                "cls": 1,
                "chair": 2,
                "book": 3,
                "medicine": 4,
                "vacuum": 5,
                "food": 6,
                "groceries": 7,
                "floor": 8,
                "mirror": 9,
                "closet/cabinet": 10,
                "doorway": 11,
                "paper/notebook": 12,
                "picture": 13,
                "phone/camera": 14,
                "sofa/couch": 15,
                "sandwich": 16,
                "cup/glass/bottle": 17,
                "towel": 18,
                "box": 19,
                "blanket": 20,
                "television": 21,
                "bag": 22,
                "refrigerator": 23,
                "table": 24,
                "light": 25,
                "broom": 26,
                "shoe": 27,
                "doorknob": 28,
                "bed": 29,
                "window": 30,
                "shelf": 31,
                "door": 32,
                "pillow": 33,
                "laptop": 34,
                "dish": 35,
                "clothes": 36,
                "person": 37,
            }
        )
        self.frame2type = (
            {
                "pad": 0,
                "start": 1,
                "regular": 2,
                "empty": 3,
                "extract": 4,
            }
            if self.dataset_name == "something"
            else {"pad": 0, "regular": 1, "extract": 2, "empty": 3}
        )


class StltDataset(Dataset):
    def __init__(self, config: StltDataConfig):
        self.config = config
        self.json_file = json.load(open(self.config.dataset_path))
        self.labels = json.load(open(self.config.labels_path))
        self.videoid2size = json.load(open(self.config.videoid2size_path))
        # Find max num objects
        max_objects = -1
        for video in self.json_file:
            for video_frame in video["frames"]:
                cur_num_objects = 0
                for frame_object in video_frame["frame_objects"]:
                    if frame_object["score"] >= self.config.score_threshold:
                        cur_num_objects += 1
                max_objects = max(max_objects, cur_num_objects)
        self.config.max_num_objects = max_objects

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx: int):
        video_id = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_id]).repeat(2)
        boxes, categories, scores, frame_types = [], [], [], []
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.config.num_frames, num_frames)
            if self.config.train
            else get_test_layout_indices(self.config.num_frames, num_frames)
        )
        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            # Prepare CLS object
            frame_types.append(
                self.config.frame2type["empty"]
                if len(frame["frame_objects"]) == 0
                else self.config.frame2type["regular"]
            )
            frame_boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0])]
            frame_categories = [self.config.category2id["cls"]]
            frame_scores = [1.0]
            # Iterate over the other objects
            for element in frame["frame_objects"]:
                if element["score"] < self.config.score_threshold:
                    continue
                # Prepare box
                box = [element["x1"], element["y1"], element["x2"], element["y2"]]
                box = fix_box(
                    box, (video_size[1].item(), video_size[0].item())
                )  # Height, Width
                box = torch.tensor(box) / video_size
                frame_boxes.append(box)
                # Prepare category
                frame_categories.append(self.config.category2id[element["category"]])
                # Prepare scores
                frame_scores.append(element["score"])
            # Ensure that everything is of the same length and pad to the max number of objects
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != self.config.max_num_objects + 1:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(0)
                frame_scores.append(0.0)
            categories.append(torch.tensor(frame_categories))
            scores.append(torch.tensor(frame_scores))
            boxes.append(torch.stack(frame_boxes, dim=0))
        # Prepare extract element
        # Boxes
        extract_box = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        extract_box[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes.append(extract_box)
        # Categories
        extract_category = torch.full((self.config.max_num_objects + 1,), 0)
        extract_category[0] = self.config.category2id["cls"]
        categories.append(extract_category)
        # Scores
        extract_score = torch.full((self.config.max_num_objects + 1,), 0.0)
        extract_score[0] = 1.0
        scores.append(extract_score)
        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(self.config.frame2type["extract"])
        # Get action(s)
        actions = self.get_actions(self.json_file[idx])

        return (
            video_id,
            torch.stack(categories, dim=0),
            torch.stack(boxes, dim=0),
            torch.stack(scores, dim=0),
            torch.tensor(frame_types),
            length,
            actions,
        )

    def get_actions(self, sample):
        if self.config.dataset_name == "something":
            actions = torch.tensor(
                int(self.labels[re.sub("[\[\]]", "", sample["template"])])
            )
        elif self.config.dataset_name == "action_genome":
            action_list = [int(action[1:]) for action in sample["actions"]]
            actions = torch.zeros(len(self.labels), dtype=torch.float)
            actions[action_list] = 1.0
        return actions


class StltCollater:
    def __init__(self, config: StltDataConfig):
        self.config = config

    def __call__(self, batch):
        (
            _,
            categories,
            boxes,
            scores,
            frame_types,
            lengths,
            labels,
        ) = zip(*batch)
        # https://github.com/pytorch/pytorch/issues/24816
        return_dict = {}
        # Pad categories
        pad_categories_tensor = torch.full((self.config.max_num_objects + 1,), 0)
        pad_categories_tensor[0] = self.config.category2id["cls"]
        return_dict["categories"] = pad_sequence(
            categories, pad_tensor=pad_categories_tensor
        )
        # Hack for padding and using scores only if the dataset is Action Genome
        if self.config.dataset_name == "action_genome":
            pad_scores_tensor = torch.full((self.config.max_num_objects + 1,), 0.0)
            pad_scores_tensor[0] = 1.0
            return_dict["scores"] = pad_sequence(scores, pad_tensor=pad_scores_tensor)
        # Pad boxes
        pad_boxes_tensor = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        pad_boxes_tensor[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        return_dict["boxes"] = pad_sequence(boxes, pad_tensor=pad_boxes_tensor)
        # Pad frame types
        return_dict["frame_types"] = pad_sequence(
            frame_types, pad_tensor=torch.tensor([self.config.frame2type["pad"]])
        )
        # Prepare lengths and labels
        return_dict["lengths"] = torch.stack([*lengths], dim=0)
        return_dict["labels"] = torch.stack([*labels], dim=0)
        # Generate mask for the padding of the boxes
        src_key_padding_mask_boxes = torch.zeros_like(
            return_dict["categories"], dtype=torch.bool
        )
        src_key_padding_mask_boxes[torch.where(return_dict["categories"] == 0)] = True
        return_dict["src_key_padding_mask_boxes"] = src_key_padding_mask_boxes
        # Generate mask for padding of the frames
        src_key_padding_mask_frames = torch.zeros(
            return_dict["frame_types"].size(), dtype=torch.bool
        )
        src_key_padding_mask_frames[
            torch.where(return_dict["frame_types"] == self.config.frame2type["pad"])
        ] = True
        return_dict["src_key_padding_mask_frames"] = src_key_padding_mask_frames

        return return_dict


# DATASETS BELLOW ARE FOR TESTING PURPOSES ONLY


class StltActionGenomeDataset(Dataset):
    def __init__(self, config: StltDataConfig):
        self.config = config
        self.json_file = json.load(open(self.config.dataset_path))
        self.labels = json.load(open(self.config.labels_path))
        self.videoid2size = json.load(open(self.config.videoid2size_path))
        # Find max num objects
        num_objects = []
        for video in self.json_file:
            for video_frame in video["frames"]:
                cur_num_objects = 0
                for frame_object in video_frame["frame_objects"]:
                    if frame_object["score"] >= self.config.score_threshold:
                        cur_num_objects += 1
                num_objects.append(cur_num_objects)
        self.config.max_num_objects = max(num_objects)

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx: int):
        video_name = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_name]).repeat(2)
        # Start loading
        boxes = []
        categories = []
        scores = []
        frame_types = []
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.config.num_frames, num_frames)
            if self.config.train
            else get_test_layout_indices(self.config.num_frames, num_frames)
        )
        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            # Prepare frame types
            frame_types.append(self.config.frame2type["regular"])
            # Prepare boxes and categories
            frame_boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0])]
            frame_categories = [self.config.category2id["cls"]]
            frame_scores = [1.0]
            for element in frame["frame_objects"]:
                if element["score"] < self.config.score_threshold:
                    continue
                # element_category = "person" if element["category"] == "person" else "object"
                element_category = element["category"]
                # element_category = element["category"]
                element_score = element["score"]
                box = [element["x1"], element["y1"], element["x2"], element["y2"]]
                box = fix_box(
                    box, (video_size[1].item(), video_size[0].item())
                )  # Height, Width
                # Prepare box
                box = torch.tensor(box) / video_size
                frame_boxes.append(box)
                # Prepare category
                frame_categories.append(self.config.category2id[element_category])
                # Prepare scores
                frame_scores.append(element_score)
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != self.config.max_num_objects + 1:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(0)
                frame_scores.append(0.0)
            categories.append(torch.tensor(frame_categories))
            scores.append(torch.tensor(frame_scores))
            boxes.append(torch.stack(frame_boxes, dim=0))

        # Prepare extract element
        extract_box = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        extract_box[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes.append(extract_box)
        # Categories
        extract_category = torch.full((self.config.max_num_objects + 1,), 0)
        extract_category[0] = self.config.category2id["cls"]
        categories.append(extract_category)
        # Scores
        extract_score = torch.full((self.config.max_num_objects + 1,), 0.0)
        extract_score[0] = 1.0
        scores.append(extract_score)
        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(self.config.frame2type["extract"])
        # Obtain video label
        action_list = [int(action[1:]) for action in self.json_file[idx]["actions"]]
        actions = torch.zeros(len(self.labels), dtype=torch.float)
        actions[action_list] = 1.0

        return (
            video_name,
            torch.stack(categories, dim=0),
            torch.stack(boxes, dim=0),
            torch.stack(scores, dim=0),
            torch.tensor(frame_types),
            length,
            actions,
        )


class StltSmthDataset(Dataset):
    def __init__(self, config: StltDataConfig):
        # Load dataset JSON file
        self.config = config
        self.json_file = json.load(open(self.config.dataset_path))
        self.labels = json.load(open(self.config.labels_path))
        self.videoid2size = json.load(open(self.config.videoid2size_path))

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx: int):
        video_id = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_id]).repeat(2)
        # Start loading
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.config.num_frames, num_frames)
            if self.config.train
            else get_test_layout_indices(self.config.num_frames, num_frames)
        )
        boxes = []
        categories = []
        frame_types = []
        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            frame_types.append(
                self.config.frame2type["empty"]
                if len(frame["frame_objects"]) == 0
                else self.config.frame2type["regular"]
            )
            frame_boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0])]
            frame_categories = [self.config.category2id["cls"]]
            for element in frame["frame_objects"]:
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
                category_name = "hand" if "hand" in element["category"] else "object"
                frame_categories.append(self.config.category2id[category_name])
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != self.config.max_num_objects + 1:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(0)
            categories.append(torch.tensor(frame_categories))
            boxes.append(torch.stack(frame_boxes, dim=0))
        # Appending extract element
        # Boxes
        extract_box = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        extract_box[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes.append(extract_box)
        boxes = torch.stack(boxes, dim=0)
        # Categories
        extract_category = torch.full((self.config.max_num_objects + 1,), 0)
        extract_category[0] = self.config.category2id["cls"]
        categories.append(extract_category)
        categories = torch.stack(categories, dim=0)
        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(self.config.frame2type["extract"])
        frame_types = torch.tensor(frame_types)
        # Obtain video label
        video_label = torch.tensor(
            int(self.labels[re.sub("[\[\]]", "", self.json_file[idx]["template"])])
        )

        return (
            video_id,
            categories,
            boxes,
            [],  # Scores
            frame_types,
            length,
            video_label,
        )
