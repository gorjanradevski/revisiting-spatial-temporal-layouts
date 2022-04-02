import argparse
import json
import os
from typing import Any, Dict, List

from natsort import natsorted
from tqdm import tqdm


def prepare_dataset(
    dataset: List[Dict[str, Any]],
    annotations: List[Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    prepared_dataset = []
    for e in tqdm(dataset):
        for i in range(len(annotations)):
            if e["id"] in annotations[i]:
                instance = {"id": e["id"], "template": e["template"], "frames": []}
                for frame in annotations[i][e["id"]]:
                    boxes = [
                        {
                            "category": "hand"
                            if "hand" in box["category"]
                            else "object",
                            "x1": box["box2d"]["x1"],
                            "y1": box["box2d"]["y1"],
                            "x2": box["box2d"]["x2"],
                            "y2": box["box2d"]["y2"],
                            "score": 1.0,
                        }
                        for box in frame["labels"]
                    ]
                    instance["frames"].append({"frame_objects": boxes})
                prepared_dataset.append(instance)

    return prepared_dataset


def create_datasets(args):
    print("Loading datasets...")
    train_dataset = json.load(open(args.train_data_path))
    val_dataset = json.load(open(os.path.join(args.val_data_path)))
    print("Loading annotations...")
    annotations = []
    for annotation_name in tqdm(natsorted(os.listdir(args.annotations_path))):
        anns = json.load(open(os.path.join(args.annotations_path, annotation_name)))
        annotations.append(anns)
    # Create and dump datasets
    print("Processing train dataset...")
    train_dataset_prepared = prepare_dataset(train_dataset, annotations)
    dump_train_dataset_path = os.path.join(args.save_data_path, "train_dataset.json")
    print(
        f"Dumping training dataset of size {len(train_dataset_prepared)} at: {dump_train_dataset_path}"
    )
    json.dump(train_dataset_prepared, open(dump_train_dataset_path, "w"))
    print("Processing validation dataset...")
    val_dataset_prepared = prepare_dataset(val_dataset, annotations)
    dump_val_dataset_path = os.path.join(args.save_data_path, "val_dataset.json")
    print(
        f"Dumping validation dataset of size {len(val_dataset_prepared)} at: {dump_val_dataset_path}"
    )
    json.dump(val_dataset_prepared, open(dump_val_dataset_path, "w"))


def main():
    parser = argparse.ArgumentParser(
        description="Creates a dataset for Something-Something and Something-Else."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the default training dataset.",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="Path to the default validation dataset.",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        required=True,
        help="From where to load annotations.",
    )
    parser.add_argument(
        "--save_data_path",
        type=str,
        default="data/",
        help="Where to save the datasets.",
    )

    args = parser.parse_args()
    create_datasets(args)


if __name__ == "__main__":
    main()
