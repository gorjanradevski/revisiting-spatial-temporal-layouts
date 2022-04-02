# Revisiting spatio-temporal layouts for compositional action recognition

[![Conference](https://img.shields.io/badge/BMVC%20Oral-2021-purple.svg?style=for-the-badge&color=f1e3ff&labelColor=purple)](https://www.bmvc2021-virtualconference.com/assets/papers/0974.pdf)    [![arXiv](https://img.shields.io/badge/arXiv-2111.01936-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2111.01936)

Codebase for [Revisiting spatio-temporal layouts for compositional action recognition](https://arxiv.org/abs/2111.01936).

## Dependencies

If you use [Poetry](https://python-poetry.org/), running ```poetry install``` inside the project should suffice.

## Preparing the data

### Something-Something and Something-Else

You need to download the [data splits and labels](https://github.com/joaanna/something_else/tree/master/code/dataset_splits), the [annotations](https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ), and the [video sizes](https://drive.google.com/file/d/1ANaDAxXoA63CA9zXalnmaskqfO4cftW4/view?usp=sharing). Make sure that the annotations for the split you want to create datasets for are in a single directory. Then, use ```create_something_datasets.py``` to create the training and test datasets as:

```python
python src/create_something_datasets.py --train_data_path "data/path-to-the-train-file.json"
                                        --val_data_path "data/path-to-the-val-file.json"
                                        --annotations_path "data/all-annotations-for-the-split/"
```

### Action-Genome

You need to download the [Action Genome data](https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys) and the [Charades data](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip). Then, use ```create_action_genome_datasets.py``` to create the training and test datasets as:

```python
python src/create_action_genome_datasets.py --action_genome_path "data/path-to-action-genome"
                                            --charades_path "data/path-to-charades"
                                            --save_datasets_path "data/directory-where-the-data-will-be-saved"
```

## Model Zoo

Trained models currently available for the Something-Else and the Action Genome dataset. If a model is not currently available and you need it, feel free to reach out as we are still in the process of releasing the models (Including Something-Something V2).

| Model | Dataset | Download |
| :--- | :--- | :--- |
| STLT | Something-Else Compositional Split Detections | [Link](https://drive.google.com/file/d/1di61ChtFeJw2fNwNvKCx7-xJrIrmd18s/view?usp=sharing) |
| LCF | Something-Else Compositional Split Detections | [Link](https://drive.google.com/file/d/1hXWiCYYINznktjzzLdj5beV50ZfXZZLV/view?usp=sharing) |
| CAF | Something-Else Compositional Split Detections | [Link](https://drive.google.com/file/d/1PV9y5ydaNLhWMtsdS5TiIUFXENE6WZ-J/view?usp=sharing)
| CACNF | Something-Else Compositional Split Detections | [Link](https://drive.google.com/file/d/1-bBLbBCOe8F-byb84cZLCwMOB1w71RTk/view?usp=sharing)
| STLT | Action Genome Oracle | [Link](https://drive.google.com/file/d/16apQ72Vpd7mt-7YC-6TT6DTvWjEC62OR/view?usp=sharing) |
| STLT | Action Genome Detections | [Link](https://drive.google.com/file/d/12WpRPW3rn9Yr3VeeCsiGBcZ4HKvBLqSa/view?usp=sharing) |

## Training and Inference

The codebase currently supports training and inference of STLT, LCF, CAF, CACNF models. Refer to the ```train.py``` and the ```inference.py``` scripts. Additonally, you need to download the Resnet3D, pretrained on Kinetics and similar from [here](https://drive.google.com/file/d/1Z1agO6kKkMr-RcQz3DTptOORrqma1dQd/view?usp=sharing), and add it in `models/`. To run inference with a trained model, e.g., STLT on Something-Else Compositional split, you can do the following:

```python
poetry run python src/inference.py --checkpoint_path "models/comp_split_detect_stlt.pt" 
                                   --test_dataset_path "data/something-somethiing/comp_split_detect/val_dataset.json"
                                   --labels_path "data/something-something/comp_split_detect/something-something-v2-labels.json"
                                   --videoid2size_path "data/something-something/videoid2size.json"
                                   --dataset_type "layout"
                                   --model_name "stlt"
                                   --dataset_name "something"
```

To run inference with a pre-trained CACNF model you can do the following:

```python
poetry run python src/inference.py --checkpoint_path "models/something-something/comp_split_detect_cacnf.pt"                                --test_dataset_path "data/something-something/comp_split_detect/val_dataset.json"                              --labels_path "data/something-something/comp_split_detect/something-something-v2-labels.json"
                                   --videoid2size_path "data/something-something/videoid2size.json" --batch_size 4 --dataset_type "multimodal"
                                   --model_name "cacnf"
                                   --dataset_name "something"
                                   --videos_path "data/something-something/dataset.hdf5"
                                   --resnet_model_path "models/something-something/r3d50_KMS_200ep.pth"
```

for both examples, make sure to provide your local paths to the dataset files and the pre-trained checkpoints.

## Citation

If you find our code useful for your own research please use the following BibTeX entry.

```tex
@article{radevski2021revisiting,
  title={Revisiting spatio-temporal layouts for compositional action recognition},
  author={Radevski, Gorjan and Moens, Marie-Francine and Tuytelaars, Tinne},
  journal={arXiv preprint arXiv:2111.01936},
  year={2021}
}
```
