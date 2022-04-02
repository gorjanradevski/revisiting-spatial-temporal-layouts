from typing import List, Tuple

import ffmpeg
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms import functional as TF


def load_video(in_filepath: str):
    """Loads a video from a filepath."""
    probe = ffmpeg.probe(in_filepath)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    out, _ = (
        ffmpeg.input(in_filepath)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        # https://github.com/kkroening/ffmpeg-python/issues/68#issuecomment-443752014
        .global_args("-loglevel", "error")
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return video


def sample_train_layout_indices(coord_nr_frames: int, nr_video_frames: int):
    # https://github.com/joaanna/something_else/blob/master/code/data_utils/data_loader_frames.py#L135
    average_duration = nr_video_frames * 1.0 / coord_nr_frames
    if average_duration > 0:
        offsets = np.multiply(
            list(range(coord_nr_frames)), average_duration
        ) + np.random.uniform(0, average_duration, size=coord_nr_frames)
        offsets = np.floor(offsets)
    elif nr_video_frames > coord_nr_frames:
        offsets = np.sort(np.random.randint(nr_video_frames, size=coord_nr_frames))
    else:
        offsets = np.arange(nr_video_frames)
    offsets = list(map(int, list(offsets)))
    return offsets


def get_test_layout_indices(coord_nr_frames: int, nr_video_frames: int):
    # https://github.com/joaanna/something_else/blob/master/code/data_utils/data_loader_frames.py#L148
    if nr_video_frames > coord_nr_frames:
        tick = nr_video_frames * 1.0 / coord_nr_frames
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(coord_nr_frames)])
    else:
        offsets = np.arange(nr_video_frames)
    offsets = list(map(int, list(offsets)))
    return offsets


def sample_appearance_indices(
    coord_nr_frames: int, nr_video_frames: int, train: bool, sample_rate=2
):
    # https://github.com/joaanna/something_else/blob/master/code/data_utils/data_loader_frames.py#L157
    d = coord_nr_frames * sample_rate  # 16 * 2
    if nr_video_frames > d:
        if train:
            # random sample
            offset = np.random.randint(0, nr_video_frames - d)
        else:
            # center crop
            offset = (nr_video_frames - d) // 2
        frame_list = list(range(offset, offset + d, sample_rate))
    else:
        # Temporal Augmentation
        if train:  # train
            if nr_video_frames - 2 < coord_nr_frames:
                # less frames than needed
                pos = np.linspace(0, nr_video_frames - 2, coord_nr_frames)
            else:  # take one
                pos = np.sort(
                    np.random.choice(
                        list(range(nr_video_frames - 2)), coord_nr_frames, replace=False
                    )
                )
        else:
            pos = np.linspace(0, nr_video_frames - 2, coord_nr_frames)
        frame_list = [round(p) for p in pos]
    # Without max(x, 0) bug when nr_video_frames = 1
    frame_list = [int(max(x, 0)) for x in frame_list]

    return frame_list


def pad_sequence(sequences: List[torch.Tensor], pad_tensor: torch.Tensor):
    num_trailing_dims = len(sequences[0].size()[1:])
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + (1,) * num_trailing_dims
    out_tensor = pad_tensor.repeat(out_dims)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor

    return out_tensor


class IdentityTransform:
    def __call__(self, image: Image):
        return image


class VideoColorJitter:
    # Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py#L1140
    def __init__(self):
        (
            self.fn_idx,
            self.brightness_factor,
            self.contrast_factor,
            self.saturation_factor,
            self.hue_factor,
        ) = ColorJitter.get_params(
            brightness=(0.75, 1.25),
            contrast=(0.75, 1.25),
            saturation=(0.75, 1.25),
            hue=(-0.1, 0.1),
        )

    def __call__(self, img: Image):
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = TF.adjust_brightness(img, self.brightness_factor)
            elif fn_id == 1 and self.contrast_factor is not None:
                img = TF.adjust_contrast(img, self.contrast_factor)
            elif fn_id == 2 and self.saturation_factor is not None:
                img = TF.adjust_saturation(img, self.saturation_factor)
            elif fn_id == 3 and self.hue_factor is not None:
                img = TF.adjust_hue(img, self.hue_factor)

        return img


class ResizeBoxes:
    # Resize boxes according to the shortest size of the image. Adapted from:
    # https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/resize_bbox.py
    def __call__(self, box: List[int], scale_factor: float):
        out_box = [e * scale_factor for e in box]

        return out_box


class CenterCropBoxes:
    # Adapted from:
    # https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/translate_bbox.py
    def __init__(self, dummy_image, spatial_size: int):
        self.left = (dummy_image.size[0] - spatial_size) // 2
        self.top = (dummy_image.size[1] - spatial_size) // 2
        self.height = spatial_size
        self.width = spatial_size

    def __call__(self, box: List[int]):
        out_box = [
            box[0] - self.left,
            box[1] - self.top,
            box[2] - self.left,
            box[3] - self.top,
        ]

        return out_box


class RandomCropBoxes:
    # Adapted from:
    # https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/translate_bbox.py
    def __init__(self, dummy_image, spatial_size):
        self.top, self.left, self.height, self.width = RandomCrop.get_params(
            dummy_image, (spatial_size, spatial_size)
        )

    def __call__(self, box: List[int]):
        out_box = [
            box[0] - self.left,
            box[1] - self.top,
            box[2] - self.left,
            box[3] - self.top,
        ]

        return out_box


def valid_box(box: List[int], frame_size: int):
    if box[0] >= frame_size and box[2] >= frame_size:
        return False
    if box[0] <= 0 and box[2] <= 0:
        return False
    if box[1] >= frame_size and box[3] >= frame_size:
        return False
    if box[1] <= 0 and box[3] <= 0:
        return False
    return True


def clamp_box(box: List[int], frame_size: int):
    out_box = [max(0, min(e, frame_size)) for e in box]
    return out_box


def fix_box(box: List[int], video_size: Tuple[int, int]):
    # Cast box elements to integers
    box = [max(0, int(b)) for b in box]
    # If x1 > x2 or y1 > y2 switch (Hack)
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]
    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]
    # Clamp to max size (Hack)
    if box[0] >= video_size[1]:
        box[0] = video_size[1] - 1
    if box[1] >= video_size[0]:
        box[1] = video_size[0] - 1
    if box[2] >= video_size[1]:
        box[2] = video_size[1] - 1
    if box[3] >= video_size[0]:
        box[3] = video_size[0] - 1
    # Fix if equal (Hack)
    if box[0] == box[2] and box[0] == 0:
        box[2] = 1
    if box[1] == box[3] and box[1] == 0:
        box[3] = 1
    if box[0] == box[2]:
        box[0] -= 1
    if box[1] == box[3]:
        box[1] -= 1
    return box
