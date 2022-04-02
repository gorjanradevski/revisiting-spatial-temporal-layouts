import argparse
import json
import os

import h5py
import numpy as np
from tqdm import tqdm


def convert_pil_to_hdf5(args):
    videoids = json.load(open(args.videoid2size_path))
    with h5py.File(args.save_hdf5_path, "a", swmr=True) as hf:
        for video_id in tqdm(videoids.keys()):
            video_path = os.path.join(args.pil_images_path, video_id)
            grp = hf.create_group(video_id)
            for frame_name in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame_name)
                with open(frame_path, "rb") as img_f:
                    binary_data = img_f.read()

                binary_data_np = np.asarray(binary_data)
                grp.create_dataset(frame_name.split(".")[0], data=binary_data_np)


def main():
    parser = argparse.ArgumentParser(description="Packs PIL images as HDF5.")
    parser.add_argument(
        "--videoid2size_path",
        type=str,
        default="data/videoid2size.json",
        help="Path to the videoid2size json file.",
    )
    parser.add_argument(
        "--pil_images_path",
        type=str,
        default="data/PIL-20bn-something-something-v2",
        help="From where to load the PIL images.",
    )
    parser.add_argument(
        "--save_hdf5_path",
        type=str,
        default="data/dataset.hdf5",
        help="Where to save the HDF5 file.",
    )
    args = parser.parse_args()
    convert_pil_to_hdf5(args)


if __name__ == "__main__":
    main()
