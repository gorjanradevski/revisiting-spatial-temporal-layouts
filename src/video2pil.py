import argparse
import json
import math
import os
import threading

from PIL import Image
from typing import List
from utils.data_utils import load_video


def read_convert_save(
    video_ids: List[str], load_videos_path: str, save_videos_path: str
):
    """Reads and converts all videos to PIL images."""
    for video_id in video_ids:
        load_video_path = os.path.join(load_videos_path, f"{video_id}.webm")
        save_video_dir = os.path.join(save_videos_path, f"{video_id}")
        if os.path.exists(save_video_dir):
            continue
        os.mkdir(save_video_dir)
        video = load_video(load_video_path)
        for i, frame in enumerate(video):
            Image.fromarray(frame).save(os.path.join(save_video_dir, f"{i}.jpg"))


def pick_and_thread(args):
    video_ids = list(json.load(open(args.videoid2size_path)))
    if not os.path.exists(args.save_videos_path):
        raise ValueError(f"{args.save_videos_path} has to exist!")
    chunk_size = math.ceil(len(video_ids) / args.num_threads)
    chunks = [
        video_ids[i : i + chunk_size] for i in range(0, len(video_ids), chunk_size)
    ]
    threads = []
    for i in range(args.num_threads):
        thread = threading.Thread(
            target=read_convert_save,
            args=(
                chunks[i],
                args.load_videos_path,
                args.save_videos_path,
            ),
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def main():
    parser = argparse.ArgumentParser(description="Saves all videos as PIL images.")
    parser.add_argument(
        "--videoid2size_path",
        type=str,
        default="data/videoid2size.json",
        help="Path to the videoid2size json file.",
    )
    parser.add_argument(
        "--load_videos_path",
        type=str,
        default="data/20bn-something-something-v2",
        help="Path to the webm videos.",
    )
    parser.add_argument(
        "--save_videos_path",
        type=str,
        default="data/PIL-20bn-something-something-v2",
        help="Path to the webm videos.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=5,
        help="How many threads to start.",
    )
    args = parser.parse_args()
    pick_and_thread(args)


if __name__ == "__main__":
    main()
